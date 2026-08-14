"""
PICO Insight Generator (Optimized)
-----------------------------------
A Streamlit application that uses MedGemma to generate PICO insights.
Features: Caching, Parallel Processing, and Robust Network Retries.
"""

import io
import os
import concurrent.futures

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

load_dotenv()

DEFAULT_MODEL_ID = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")
HF_API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model_id}"
REQUEST_TIMEOUT_SECONDS = 60
MAX_CONCURRENT_REQUESTS = 5  # Speed up file uploads by processing in parallel

PICO_SYSTEM_PROMPT = """You are a clinical evidence assistant helping a health \
professional interpret a clinical question using the PICO framework.

Given the clinical query below, respond with a concise, clearly structured \
natural-language insight that:
1. Identifies the Population/Patient/Problem (P)
2. Identifies the Intervention or Exposure (I)
3. Identifies the Comparison or Control (C), if one is implied or stated
4. Identifies the Outcome(s) of interest (O)
5. Closes with a 2-3 sentence plain-language clinical insight synthesizing \
the PICO elements into a coherent answer or research framing.

If any PICO element is not present in the query, explicitly note that it is \
"Not specified" rather than guessing. Do not provide medical advice intended \
to replace clinical judgement; frame the response as a structured evidence \
summary aid.

Clinical query:
\"\"\"{query}\"\"\"

Respond using this exact structure:

**Population/Problem (P):** ...
**Intervention/Exposure (I):** ...
**Comparison (C):** ...
**Outcome (O):** ...
**Clinical Insight:** ...
"""

# --------------------------------------------------------------------------
# Network & API Helper Functions
# --------------------------------------------------------------------------

@st.cache_resource
def get_retry_session() -> requests.Session:
    """
    Creates a robust requests Session that automatically handles DNS errors, 
    connection timeouts, and HuggingFace rate limits/cold starts (429, 503).
    """
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,  # Waits 2s, 4s, 8s, 16s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_api_token() -> str | None:
    """Fetch the Hugging Face API token from the environment / Streamlit secrets."""
    try:
        if "HF_API_TOKEN" in st.secrets:
            return st.secrets["HF_API_TOKEN"]
    except Exception:
        pass
    return os.getenv("HF_API_TOKEN")

def build_prompt(query_text: str) -> str:
    return PICO_SYSTEM_PROMPT.format(query=query_text.strip())

# The underscore in _api_token tells Streamlit NOT to hash the secret key for caching
@st.cache_data(show_spinner=False, max_entries=500)
def call_medgemma(query_text: str, _api_token: str, model_id: str) -> str:
    """
    Calls the Hugging Face Inference API. Cached by Streamlit to prevent
    duplicate API calls for identical queries.
    """
    if not query_text or not query_text.strip():
        raise ValueError("Cannot analyze an empty query.")
    if not _api_token:
        raise RuntimeError("No Hugging Face API token configured.")

    url = HF_API_URL_TEMPLATE.format(model_id=model_id)
    headers = {
        "Authorization": f"Bearer {_api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": build_prompt(query_text),
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }

    session = get_retry_session()
    
    try:
        response = session.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Network error (e.g., DNS/Connection failure): {exc}")

    # Handle specific HTTP errors safely
    if response.status_code == 401:
        raise RuntimeError("Invalid or expired Hugging Face API key (401).")
    if response.status_code == 403:
        raise RuntimeError(f"Access denied (403) to gated model '{model_id}'.")
    if response.status_code == 404:
        raise RuntimeError(f"Model '{model_id}' not found or not on the free API.")
    if response.status_code == 429:
        raise RuntimeError("Rate limit exceeded (429) even after retries. Upgrade HF plan.")
    if not response.ok:
        raise RuntimeError(f"API request failed ({response.status_code}): {response.text[:200]}")

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError("Received a non-JSON response from the API.")

    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"API Error: {data['error']}")
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()

    raise RuntimeError(f"Unexpected API response format: {data}")

# --------------------------------------------------------------------------
# Data Processing Helpers
# --------------------------------------------------------------------------

def load_queries_from_upload(uploaded_file) -> pd.DataFrame:
    """Parses CSV or TXT files into a standardized DataFrame."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            raise ValueError(f"Could not parse CSV file: {exc}")

        if df.empty:
            raise ValueError("The uploaded CSV file contains no rows.")

        candidate_cols = [c for c in df.columns if str(c).strip().lower() in 
                          ("query", "question", "clinical_query", "text")]
        col = candidate_cols[0] if candidate_cols else df.columns[0]
        
        queries = df[col].dropna().astype(str).str.strip()
        queries = queries[queries.str.len() > 0]
        
        if queries.empty:
            raise ValueError("No usable query text was found in the CSV.")
        return pd.DataFrame({"query": queries.reset_index(drop=True)})

    elif filename.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            raise ValueError("The uploaded text file is empty.")
        return pd.DataFrame({"query": lines})

    raise ValueError("Unsupported file type. Use .csv or .txt.")

def results_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="PICO Insight Generator", page_icon="🩺", layout="centered")

st.title("🩺 PICO Insight Generator")
st.caption("Upload clinical queries to generate structured PICO insights powered by MedGemma.")

with st.sidebar:
    st.header("⚙️ Settings")
    model_id = st.text_input("MedGemma model ID", value=DEFAULT_MODEL_ID)
    api_token = get_api_token()
    
    if api_token:
        st.success("API key detected.", icon="✅")
    else:
        st.error("No API key found. Set HF_API_TOKEN.", icon="⚠️")

    st.markdown("---")
    st.markdown("**About**\n\nThis tool is an evidence-summary aid. It does **not** replace clinical judgement.")

tab_upload, tab_manual = st.tabs(["📁 Upload File", "✍️ Type a Query"])
results_placeholder = st.container()

# ---- Tab 1: File upload (Parallelized) -----------------------------------
with tab_upload:
    st.write("Accepted formats: `.csv` (with a `query` column) or `.txt`")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            queries_df = load_queries_from_upload(uploaded_file)
            st.success(f"Loaded {len(queries_df)} query(ies).")
            st.dataframe(queries_df.head(3), use_container_width=True, hide_index=True)

            if st.button("🔎 Analyze Queries", type="primary", key="analyze_file"):
                if not api_token:
                    st.error("No valid Hugging Face API key configured.")
                    st.stop()

                progress = st.progress(0.0, text="Starting parallel analysis...")
                results_list = [None] * len(queries_df) # Preserve original order

                # Process API requests in parallel using ThreadPoolExecutor
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                    # Submit all tasks
                    future_to_index = {
                        executor.submit(call_medgemma, row["query"], api_token, model_id): i
                        for i, row in queries_df.iterrows()
                    }

                    completed = 0
                    for future in concurrent.futures.as_completed(future_to_index):
                        idx = future_to_index[future]
                        original_query = queries_df.iloc[idx]["query"]
                        
                        try:
                            insight = future.result()
                            results_list[idx] = {"query": original_query, "pico_insight": insight, "status": "success"}
                        except Exception as exc:
                            results_list[idx] = {"query": original_query, "pico_insight": str(exc), "status": "error"}
                        
                        completed += 1
                        progress.progress(completed / len(queries_df), text=f"Analyzed {completed} of {len(queries_df)}...")

                progress.empty()
                results_df = pd.DataFrame(results_list)
                
                with results_placeholder:
                    st.subheader("Results")
                    for _, row in results_df.iterrows():
                        icon = "✅" if row["status"] == "success" else "❌"
                        with st.expander(f"{icon} {row['query'][:80]}..."):
                            st.markdown(row["pico_insight"])

                    st.download_button(
                        "⬇️ Download results as CSV",
                        data=results_to_csv_bytes(results_df),
                        file_name="pico_insights.csv",
                        mime="text/csv"
                    )
        except ValueError as exc:
            st.error(f"File error: {exc}")

# ---- Tab 2: Manual entry -------------------------------------------------
with tab_manual:
    manual_query = st.text_area(
        "Clinical query",
        placeholder="e.g. In adults with type 2 diabetes, does metformin...",
        height=120,
    )

    if st.button("🔎 Analyze Query", type="primary", key="analyze_manual"):
        if not api_token:
            st.error("No valid Hugging Face API key configured.")
        elif not manual_query.strip():
            st.warning("Please enter a query before analyzing.")
        else:
            with st.spinner("Analyzing query with MedGemma..."):
                try:
                    insight = call_medgemma(manual_query, api_token, model_id)
                    with results_placeholder:
                        st.subheader("Result")
                        st.markdown(insight)
                except Exception as exc:
                    st.error(f"Error: {exc}")

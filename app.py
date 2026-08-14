"""
PICO Insight Generator
-----------------------
A Streamlit application that lets a health professional upload one or more
clinical queries and uses a MedGemma model (via the Hugging Face Inference
API) to generate a natural-language insight structured around the PICO
framework (Population, Intervention, Comparison, Outcome).

Run locally with:
    streamlit run app.py
"""

import io
import os
import time

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

load_dotenv()  # Loads variables from a local .env file (never committed to git)

# You can swap this for any MedGemma checkpoint you have access to on the Hub,
# e.g. "google/medgemma-4b-it" or "google/medgemma-27b-text-it".
DEFAULT_MODEL_ID = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")
HF_API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model_id}"
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5  # used mainly when the model is "cold" (loading)

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
# Helper functions
# --------------------------------------------------------------------------


def get_api_token() -> str | None:
    """Fetch the Hugging Face API token from the environment / Streamlit secrets."""
    # Prefer Streamlit secrets when deployed on Streamlit Community Cloud,
    # fall back to a local .env-managed environment variable otherwise.
    token = None
    try:
        token = st.secrets.get("HF_API_TOKEN")  # type: ignore[attr-defined]
    except Exception:
        token = None
    if not token:
        token = os.getenv("HF_API_TOKEN")
    return token


def build_prompt(query_text: str) -> str:
    return PICO_SYSTEM_PROMPT.format(query=query_text.strip())


def call_medgemma(query_text: str, api_token: str, model_id: str) -> str:
    """
    Calls the Hugging Face Inference API for the given MedGemma model and
    returns the generated PICO insight as plain text.

    Raises a RuntimeError with a human-readable message on failure, so the
    caller can display it cleanly in the UI instead of crashing.
    """
    if not query_text or not query_text.strip():
        raise ValueError("Cannot analyze an empty query.")

    if not api_token or not api_token.strip():
        raise RuntimeError(
            "No Hugging Face API token found. Please set HF_API_TOKEN in your "
            ".env file (local) or in Streamlit secrets (deployed app)."
        )

    url = HF_API_URL_TEMPLATE.format(model_id=model_id)
    headers = {
        "Authorization": f"Bearer {api_token}",
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

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except requests.exceptions.Timeout as exc:
            last_error = f"Request timed out (attempt {attempt}/{MAX_RETRIES})."
            time.sleep(RETRY_BACKOFF_SECONDS)
            continue
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Network error while contacting the API: {exc}")

        if response.status_code == 401:
            raise RuntimeError(
                "Invalid or expired Hugging Face API key (401 Unauthorized). "
                "Please check your HF_API_TOKEN."
            )
        if response.status_code == 403:
            raise RuntimeError(
                "Access denied (403). Your token may not have access to this "
                f"gated model ('{model_id}'). Request access on the model's "
                "Hugging Face page and accept its usage terms."
            )
        if response.status_code == 404:
            raise RuntimeError(
                f"Model '{model_id}' was not found, or is not deployed on the "
                "free Inference API. Verify the model ID."
            )
        if response.status_code == 503:
            # Model is loading (cold start) - worth retrying.
            last_error = "The model is still loading on Hugging Face's servers."
            time.sleep(RETRY_BACKOFF_SECONDS)
            continue
        if response.status_code == 429:
            raise RuntimeError(
                "Rate limit exceeded (429). Please wait a moment before trying "
                "again, or upgrade your Hugging Face plan for higher limits."
            )
        if not response.ok:
            raise RuntimeError(
                f"API request failed with status {response.status_code}: "
                f"{response.text[:300]}"
            )

        try:
            data = response.json()
        except ValueError:
            raise RuntimeError("Received a non-JSON response from the API.")

        # The Inference API can return either a list of generations or a dict
        # with an "error" key if something went wrong server-side.
        if isinstance(data, dict) and "error" in data:
            last_error = data["error"]
            time.sleep(RETRY_BACKOFF_SECONDS)
            continue

        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()

        raise RuntimeError(f"Unexpected API response format: {data}")

    raise RuntimeError(
        f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


def load_queries_from_upload(uploaded_file) -> pd.DataFrame:
    """
    Accepts a .csv (with a 'query' column) or a .txt file (one query per
    line) and returns a DataFrame with a single 'query' column.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
        except pd.errors.EmptyDataError:
            raise ValueError("The uploaded CSV file is empty.")
        except Exception as exc:
            raise ValueError(f"Could not parse CSV file: {exc}")

        if df.empty:
            raise ValueError("The uploaded CSV file contains no rows.")

        # Try to find a sensible column name for the query text.
        candidate_cols = [c for c in df.columns if c.strip().lower() in
                           ("query", "question", "clinical_query", "text")]
        if candidate_cols:
            col = candidate_cols[0]
        else:
            col = df.columns[0]  # fall back to first column

        queries = df[col].dropna().astype(str).str.strip()
        queries = queries[queries.str.len() > 0]
        if queries.empty:
            raise ValueError("No usable query text was found in the CSV file.")
        return pd.DataFrame({"query": queries.reset_index(drop=True)})

    elif filename.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            raise ValueError("The uploaded text file is empty.")
        return pd.DataFrame({"query": lines})

    else:
        raise ValueError(
            "Unsupported file type. Please upload a .csv (with a 'query' "
            "column) or a .txt file (one query per line)."
        )


def results_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="PICO Insight Generator",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 PICO Insight Generator")
st.caption(
    "Upload clinical queries and generate structured PICO (Population, "
    "Intervention, Comparison, Outcome) insights powered by MedGemma."
)

with st.sidebar:
    st.header("⚙️ Settings")
    model_id = st.text_input("MedGemma model ID", value=DEFAULT_MODEL_ID)

    api_token = get_api_token()
    if api_token:
        st.success("Hugging Face API key detected.", icon="✅")
    else:
        st.error(
            "No API key found. Set HF_API_TOKEN in a local .env file, or in "
            "Streamlit secrets if deployed.",
            icon="⚠️",
        )

    st.markdown("---")
    st.markdown(
        "**About**\n\n"
        "This tool is a structured evidence-summary aid for health "
        "professionals. It does **not** replace clinical judgement or "
        "provide direct medical advice."
    )

tab_upload, tab_manual = st.tabs(["📁 Upload File", "✍️ Type a Query"])

results_placeholder = st.container()

# ---- Tab 1: File upload -------------------------------------------------
with tab_upload:
    st.subheader("Upload a query file")
    st.write(
        "Accepted formats: `.csv` (with a `query` column) or `.txt` "
        "(one clinical query per line)."
    )
    uploaded_file = st.file_uploader(
        "Choose a file", type=["csv", "txt"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        try:
            queries_df = load_queries_from_upload(uploaded_file)
            st.success(f"Loaded {len(queries_df)} quer{'y' if len(queries_df)==1 else 'ies'}.")
            st.dataframe(queries_df, use_container_width=True, hide_index=True)

            if st.button("🔎 Analyze Queries", type="primary", key="analyze_file"):
                if not api_token:
                    st.error("Cannot analyze: no valid Hugging Face API key configured.")
                else:
                    output_rows = []
                    progress = st.progress(0.0, text="Starting analysis...")
                    for i, row in queries_df.iterrows():
                        progress.progress(
                            (i + 1) / len(queries_df),
                            text=f"Analyzing query {i + 1} of {len(queries_df)}...",
                        )
                        try:
                            insight = call_medgemma(row["query"], api_token, model_id)
                            output_rows.append(
                                {"query": row["query"], "pico_insight": insight, "status": "success"}
                            )
                        except (ValueError, RuntimeError) as exc:
                            output_rows.append(
                                {"query": row["query"], "pico_insight": str(exc), "status": "error"}
                            )
                    progress.empty()

                    results_df = pd.DataFrame(output_rows)
                    with results_placeholder:
                        st.subheader("Results")
                        for _, row in results_df.iterrows():
                            icon = "✅" if row["status"] == "success" else "❌"
                            with st.expander(f"{icon} {row['query'][:80]}"):
                                st.markdown(row["pico_insight"])

                        st.download_button(
                            "⬇️ Download results as CSV",
                            data=results_to_csv_bytes(results_df),
                            file_name="pico_insights.csv",
                            mime="text/csv",
                        )
        except ValueError as exc:
            st.error(f"File error: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface unexpected errors to the UI
            st.error(f"Unexpected error while processing the file: {exc}")

# ---- Tab 2: Manual entry -------------------------------------------------
with tab_manual:
    st.subheader("Type a single clinical query")
    manual_query = st.text_area(
        "Clinical query",
        placeholder=(
            "e.g. In adults with type 2 diabetes, does metformin monotherapy "
            "reduce HbA1c more than lifestyle modification alone over 6 months?"
        ),
        height=120,
    )

    if st.button("🔎 Analyze Query", type="primary", key="analyze_manual"):
        if not api_token:
            st.error("Cannot analyze: no valid Hugging Face API key configured.")
        elif not manual_query.strip():
            st.warning("Please enter a query before analyzing.")
        else:
            with st.spinner("Analyzing query with MedGemma..."):
                try:
                    insight = call_medgemma(manual_query, api_token, model_id)
                    with results_placeholder:
                        st.subheader("Result")
                        st.markdown(insight)
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unexpected error: {exc}")

st.markdown("---")
st.caption(
    "⚠️ This tool provides structured evidence-summary assistance only. "
    "It is not a substitute for professional clinical judgement."
)

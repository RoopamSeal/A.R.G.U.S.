"""
PICO Insight Generator
-----------------------
A Streamlit application that lets a health professional upload one or more
clinical queries and uses an LLM to generate a natural-language insight
structured around the PICO framework (Population, Intervention, Comparison,
Outcome).

Note: MedGemma is not hosted on Hugging Face's free serverless router as of
2026. By default this app calls a free general-purpose instruct model via
that router. To use actual MedGemma, deploy it to a paid HF Inference
Endpoint or Vertex AI Model Garden and enter that endpoint's URL in the
sidebar ("Custom endpoint" mode).

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

# NOTE: As of 2026, MedGemma is NOT deployed on Hugging Face's free serverless
# "Inference Providers" router. To actually call MedGemma you need either a
# paid HF Inference Endpoint or a Vertex AI Model Garden deployment, and you
# provide its base URL below (CUSTOM_ENDPOINT_URL / sidebar field).
# For a free, zero-setup demo, this app falls back to a general-purpose
# instruct model that IS live on the free router.
DEFAULT_FREE_MODEL_ID = os.getenv("FREE_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
DEFAULT_CUSTOM_ENDPOINT_URL = os.getenv("CUSTOM_ENDPOINT_URL", "")  # e.g. a paid HF Endpoint base URL

# Hugging Face retired api-inference.huggingface.co in favor of the unified
# "Inference Providers" router, which speaks the OpenAI-compatible chat
# completions format.
HF_ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
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


def resolve_chat_url(custom_endpoint_url: str) -> str:
    """
    Decides which base URL to call:
    - If a custom endpoint (e.g. a paid HF Inference Endpoint or any other
      OpenAI-compatible server) is configured, use it.
    - Otherwise, fall back to Hugging Face's free serverless router.
    """
    if custom_endpoint_url and custom_endpoint_url.strip():
        base = custom_endpoint_url.strip().rstrip("/")
        # Dedicated HF Inference Endpoints (and most OpenAI-compatible
        # servers) expose chat completions at /v1/chat/completions.
        if base.endswith("/v1/chat/completions"):
            return base
        return f"{base}/v1/chat/completions"
    return HF_ROUTER_CHAT_URL


def call_medgemma(
    query_text: str,
    api_token: str,
    model_id: str,
    custom_endpoint_url: str = "",
) -> str:
    """
    Calls an OpenAI-compatible chat completions endpoint (either Hugging
    Face's free Inference Providers router, or a custom/dedicated endpoint
    such as a paid HF Inference Endpoint) and returns the generated PICO
    insight as plain text.

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

    url = resolve_chat_url(custom_endpoint_url)
    using_custom_endpoint = bool(custom_endpoint_url and custom_endpoint_url.strip())

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": build_prompt(query_text)},
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except requests.exceptions.Timeout:
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
            if using_custom_endpoint:
                raise RuntimeError(
                    f"404 from your custom endpoint ({url}). Double-check the "
                    "endpoint URL is correct and running."
                )
            raise RuntimeError(
                f"Model '{model_id}' was not found on Hugging Face's free "
                "router. Many specialized models (including MedGemma) are not "
                "hosted on the free tier — you'll need a paid Inference "
                "Endpoint or Vertex AI deployment, entered as a custom "
                "endpoint URL, to use that model."
            )
        if response.status_code == 503:
            # Model is loading (cold start) - worth retrying.
            last_error = "The model is still loading on the server."
            time.sleep(RETRY_BACKOFF_SECONDS)
            continue
        if response.status_code == 429:
            raise RuntimeError(
                "Rate limit exceeded (429). Please wait a moment before trying "
                "again, or upgrade your plan for higher limits."
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

        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            last_error = err_msg
            time.sleep(RETRY_BACKOFF_SECONDS)
            continue

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
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

    inference_mode = st.radio(
        "Inference source",
        ["Free hosted model (HF router)", "Custom endpoint (paid / dedicated)"],
        help=(
            "MedGemma is not available on Hugging Face's free serverless "
            "router. Use 'Free hosted model' to try the app with a "
            "general-purpose instruct model at no cost, or 'Custom endpoint' "
            "to point at a paid HF Inference Endpoint / Vertex AI deployment "
            "actually running MedGemma."
        ),
    )

    custom_endpoint_url = ""
    if inference_mode == "Custom endpoint (paid / dedicated)":
        custom_endpoint_url = st.text_input(
            "Endpoint base URL",
            value=DEFAULT_CUSTOM_ENDPOINT_URL,
            placeholder="https://your-endpoint-id.endpoints.huggingface.cloud",
            help="Base URL of your dedicated Inference Endpoint (no trailing /v1/...).",
        )
        model_id = st.text_input(
            "Model ID (as configured on your endpoint)",
            value="google/medgemma-4b-it",
        )
        if not custom_endpoint_url.strip():
            st.warning("Enter your endpoint URL to use this mode.", icon="⚠️")
    else:
        model_id = st.text_input(
            "Free model ID (hosted on HF router)",
            value=DEFAULT_FREE_MODEL_ID,
            help="Any instruct model available via Hugging Face's free Inference Providers router.",
        )
        st.caption(
            "ℹ️ This is a general-purpose model, not medically fine-tuned. "
            "Switch to 'Custom endpoint' once you have MedGemma deployed "
            "somewhere with GPU access."
        )

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
                            insight = call_medgemma(
                                row["query"], api_token, model_id, custom_endpoint_url
                            )
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
                    insight = call_medgemma(
                        manual_query, api_token, model_id, custom_endpoint_url
                    )
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

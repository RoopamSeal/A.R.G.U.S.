"""
Configuration for the Evidence Insight Repository.
Loads API keys and settings from environment variables (.env file).

v3: scoped down to only Module 1 (Disease & burden) and Module 7
(Real-world evidence) - the other 5 modules from the framework are dropped.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Groq API key - get a free one at https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# NCBI/PubMed requires an email to identify API requests (no key needed for light use)
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "")

# Optional: an NCBI API key raises your rate limit from 3 to 10 requests/second.
# Get one free at https://www.ncbi.nlm.nih.gov/account/settings/
ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY", "")

# Pipeline defaults
GROQ_MODEL = "openai/gpt-oss-120b"
MAX_ABSTRACTS_PER_QUERY = 8
CACHE_FILE = "insight_cache.json"
ABSTRACTS_PER_CATEGORY = 3  # 5 categories x 3 = up to 15 abstracts per topic

# How many abstracts to send to the LLM per extract_insights() call. If your
# insight_llm.py batches abstracts (rather than sending all of them in one
# request), this caps each batch so the prompt doesn't get too large.
# NOTE: restored after being dropped by a full config.py rewrite - confirm
# this value (and that this is the only place it's used) against your
# current insight_llm.py.
LLM_BATCH_SIZE = 5

# --- Module 1: Disease, burden, treatment landscape and unmet need ---
MODULE_1_CATEGORIES = {
    "disease_definition": {
        "label": "Disease definition & course",
        "color": "#0B5ED7",
        "search_terms": "definition classification pathophysiology diagnosis clinical course",
    },
    "epidemiology": {
        "label": "Epidemiology",
        "color": "#0B5ED7",
        "search_terms": "epidemiology prevalence incidence mortality risk factors",
    },
    "patient_funnel": {
        "label": "Patient funnel",
        "color": "#0B5ED7",
        "search_terms": "diagnosis rate treatment rate patient pathway referral",
    },
    "burden_of_disease": {
        "label": "Burden of disease",
        "color": "#0B5ED7",
        "search_terms": "disease burden morbidity mortality quality of life economic burden",
    },
    "unmet_need": {
        "label": "Unmet need",
        "color": "#0B5ED7",
        "search_terms": "unmet need treatment gap",
    },
}

# --- Module 7: Real-world evidence ---
MODULE_7_CATEGORIES = {
    "data_source_inventory": {
        "label": "Data source inventory",
        "color": "#5C6672",
        "search_terms": "registries claims database electronic health records real-world data source",
    },
    "natural_history": {
        "label": "Natural history",
        "color": "#5C6672",
        "search_terms": "natural history disease progression prognostic factors untreated patients",
    },
    "treatment_patterns": {
        "label": "Treatment patterns",
        "color": "#5C6672",
        "search_terms": "real-world treatment patterns initiation switching discontinuation adherence persistence",
    },
    "outcomes_burden": {
        "label": "Outcomes & burden",
        "color": "#5C6672",
        "search_terms": "real-world effectiveness safety healthcare resource utilization cost caregiver burden",
    },
    "comparative_options": {
        "label": "Comparative options",
        "color": "#5C6672",
        "search_terms": "real-world comparative effectiveness external control synthetic control confounding",
    },
}


def validate_config():
    """Raise a clear error early if required keys are missing, instead of a
    confusing traceback later when Entrez or Groq gets called."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not ENTREZ_EMAIL:
        missing.append("ENTREZ_EMAIL")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"Copy .env.example to .env and fill them in."
        )

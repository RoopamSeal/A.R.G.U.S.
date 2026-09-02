"""
Configuration for the PubMed Insight Generator.
Loads API keys and settings from environment variables (.env file).
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
LLM_BATCH_SIZE = 5  # abstracts per LLM call - keeps responses well under the token limit
LLM_MAX_TOKENS = 4096

# --- v2: Module 1 (Disease, burden, treatment landscape and unmet need) ---
# Each sub-category gets its own targeted PubMed search, so the app can group
# results the way the evidence framework does, instead of one generic bucket.
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

# All 7 modules from the framework - only Module 1 is built so far
ALL_MODULES = [
    "Disease & burden",
    "Competitive",
    "Regulatory",
    "Trials",
    "HTA",
    "Pricing",
    "RWE",
]

ABSTRACTS_PER_CATEGORY = 3  # 5 categories x 3 = up to 15 abstracts per topic


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

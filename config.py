"""
Configuration for the Evidence Insight Repository.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "")
ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY", "")

GROQ_MODEL = "openai/gpt-oss-120b"
MAX_ABSTRACTS_PER_QUERY = 8
CACHE_FILE = "insight_cache.json"
ABSTRACTS_PER_CATEGORY = 3

LLM_BATCH_SIZE = 3
LLM_MAX_TOKENS = 3000
LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_BASE_DELAY = 8

# search_terms are now lists of alternative phrases, OR'd together in the
# query instead of AND'd as literal words - this was silently starving
# recall (especially on Module 7/RWE, whose terms are more compound).
MODULE_1_CATEGORIES = {
    "disease_definition": {
        "label": "Disease definition & course", "color": "#0B5ED7",
        "search_terms": ["definition", "classification", "pathophysiology", "diagnosis", "clinical course"],
    },
    "epidemiology": {
        "label": "Epidemiology", "color": "#0B5ED7",
        "search_terms": ["epidemiology", "prevalence", "incidence", "mortality", "risk factors"],
    },
    "patient_funnel": {
        "label": "Patient funnel", "color": "#0B5ED7",
        "search_terms": ["diagnosis rate", "treatment rate", "patient pathway", "referral"],
    },
    "burden_of_disease": {
        "label": "Burden of disease", "color": "#0B5ED7",
        "search_terms": ["disease burden", "morbidity", "mortality", "quality of life", "economic burden"],
    },
    "unmet_need": {
        "label": "Unmet need", "color": "#0B5ED7",
        "search_terms": ["unmet need", "treatment gap"],
    },
}

MODULE_7_CATEGORIES = {
    "data_source_inventory": {
        "label": "Data source inventory", "color": "#5C6672",
        "search_terms": ["registry", "claims database", "electronic health records", "real-world data"],
    },
    "natural_history": {
        "label": "Natural history", "color": "#5C6672",
        "search_terms": ["natural history", "disease progression", "prognostic factors", "untreated"],
    },
    "treatment_patterns": {
        "label": "Treatment patterns", "color": "#5C6672",
        "search_terms": ["treatment patterns", "treatment initiation", "treatment switching", "discontinuation", "adherence", "persistence"],
    },
    "outcomes_burden": {
        "label": "Outcomes & burden", "color": "#5C6672",
        "search_terms": ["real-world effectiveness", "real-world safety", "healthcare resource utilization", "cost of illness", "caregiver burden"],
    },
    "comparative_options": {
        "label": "Comparative options", "color": "#5C6672",
        "search_terms": ["comparative effectiveness", "external control", "synthetic control", "propensity score", "confounding by indication"],
    },
}


def validate_config():
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

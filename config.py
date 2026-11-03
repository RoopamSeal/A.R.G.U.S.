"""
A.R.G.U.S. configuration: molecule- and indication-agnostic Market Access workspace.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "")
ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY", "")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_ABSTRACTS_PER_QUERY = 8
CACHE_FILE = os.getenv("ARGUS_CACHE_FILE", "insight_cache.json")
ABSTRACTS_PER_CATEGORY = 3
LLM_BATCH_SIZE = 3
LLM_MAX_TOKENS = 3500
LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_BASE_DELAY = 8

MODULES = {
    "disease_burden": {
        "number": 1, "label": "Disease & burden",
        "caption": "Disease context, epidemiology, patient funnel, burden and unmet need.",
        "categories": {
            "disease_definition": ["definition", "classification", "pathophysiology", "diagnosis", "clinical course"],
            "epidemiology": ["epidemiology", "prevalence", "incidence", "mortality", "risk factors"],
            "patient_funnel": ["diagnosis rate", "treatment rate", "patient pathway", "referral"],
            "burden_of_disease": ["disease burden", "morbidity", "quality of life", "economic burden"],
            "unmet_need": ["unmet need", "treatment gap", "limitations of current treatment"],
            "standard_of_care": ["standard of care", "clinical guideline", "treatment pathway"],
        },
    },
    "competitive": {
        "number": 2, "label": "Competitive landscape",
        "caption": "Marketed therapies, pipeline, differentiation, future comparators and competitive watch.",
        "categories": {
            "marketed_therapies": ["approved treatment", "standard treatment", "comparative efficacy", "treatment utilization"],
            "pipeline": ["clinical trial", "pipeline", "investigational therapy", "phase 2", "phase 3"],
            "differentiation": ["comparative effectiveness", "head-to-head", "safety", "patient preference"],
            "future_comparator": ["treatment sequencing", "combination therapy", "biosimilar", "generic"],
        },
    },
    "regulatory": {
        "number": 3, "label": "Regulatory & label",
        "caption": "Regulatory precedent, label scope, special designations and payer evidence bridge.",
        "categories": {
            "approval_precedent": ["regulatory approval", "FDA", "EMA", "marketing authorization"],
            "label_population": ["product label", "indication", "patient population", "contraindication"],
            "regulatory_pathways": ["accelerated approval", "conditional approval", "orphan designation"],
            "regulatory_hta_bridge": ["regulatory endpoint", "surrogate endpoint", "health technology assessment"],
        },
    },
    "trials_endpoints": {
        "number": 4, "label": "Trials & endpoints",
        "caption": "Pivotal trial library, endpoint map, comparative feasibility, PICO and JCA readiness.",
        "categories": {
            "pivotal_trials": ["randomized controlled trial", "pivotal trial", "clinical trial design"],
            "endpoints": ["primary endpoint", "patient reported outcome", "quality of life", "utility"],
            "comparative_feasibility": ["network meta-analysis", "indirect treatment comparison", "matching adjusted indirect comparison"],
            "pico_jca": ["PICO", "joint clinical assessment", "HTA evidence", "comparative evidence"],
        },
    },
    "hta_reimbursement": {
        "number": 5, "label": "HTA & reimbursement",
        "caption": "HTA outcomes, reimbursement conditions, access channels and evidence requirements.",
        "categories": {
            "hta_decisions": ["health technology assessment", "reimbursement recommendation", "cost effectiveness"],
            "access_conditions": ["reimbursement criteria", "coverage restriction", "managed entry agreement"],
            "payer_evidence": ["payer evidence", "budget impact", "relative effectiveness"],
            "us_access": ["formulary", "prior authorization", "Medicare", "Medicaid"],
        },
    },
    "pricing": {
        "number": 6, "label": "Pricing & value corridor",
        "caption": "Public price benchmarks, normalized treatment costs, price corridor and early health economics.",
        "categories": {
            "price_benchmarks": ["drug price", "list price", "pharmaceutical pricing", "price comparison"],
            "economic_value": ["cost effectiveness", "economic evaluation", "cost utility analysis"],
            "budget_impact": ["budget impact analysis", "healthcare costs", "resource utilization"],
            "pricing_dynamics": ["international reference pricing", "price negotiation", "generic competition"],
        },
    },
    "rwe": {
        "number": 7, "label": "Real-world evidence",
        "caption": "RWD inventory, natural history, treatment patterns, outcomes, comparative options and study readiness.",
        "categories": {
            "data_source_inventory": ["registry", "claims database", "electronic health records", "real-world data"],
            "natural_history": ["natural history", "disease progression", "prognostic factors", "untreated"],
            "treatment_patterns": ["treatment patterns", "treatment initiation", "treatment switching", "discontinuation", "adherence", "persistence"],
            "outcomes_burden": ["real-world effectiveness", "real-world safety", "healthcare resource utilization", "cost of illness"],
            "comparative_options": ["comparative effectiveness", "external control", "synthetic control", "propensity score"],
            "rwe_readiness": ["data quality", "data linkage", "observational study", "target trial emulation"],
        },
    },
}

MODULE_COLORS = {
    "disease_burden": "#2563EB", "competitive": "#0F766E",
    "regulatory": "#7C3AED", "trials_endpoints": "#B45309",
    "hta_reimbursement": "#BE185D", "pricing": "#475569", "rwe": "#0F766E",
}

EVIDENCE_STATUSES = ["Verified evidence", "AI synthesis/interpretation", "Forecast assumption", "Evidence gap"]
CONFIDENCE_LEVELS = ["High", "Moderate", "Low"]

def validate_config():
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not ENTREZ_EMAIL:
        missing.append("ENTREZ_EMAIL")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Add them to your deployment secrets or .env file."
        )

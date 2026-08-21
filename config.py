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
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_ABSTRACTS_PER_QUERY = 8
CACHE_FILE = "insight_cache.json"


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

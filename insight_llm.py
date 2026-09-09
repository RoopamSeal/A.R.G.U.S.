"""
Sends PubMed abstracts to an LLM (Groq) and gets back structured insights.
"""
import json
import re
import time
from groq import Groq
import config

client = Groq(api_key=config.GROQ_API_KEY)

SYSTEM_PROMPT = """You are a biomedical evidence analyst building a structured evidence \
repository. You will be given a list of PubMed abstracts, each tagged with a "category" \
and a "pmid".

For EACH abstract, extract a structured insight. Respond ONLY with a JSON array (no \
markdown, no preamble, no code fences). Each element must have exactly these fields:

- "pmid": echo back the pmid, as a string
- "category": echo back the category, exactly as given
- "pico": object with "population", "intervention", "comparison", "outcome". For \
  categories describing real-world/observational evidence (data source inventory, \
  natural history, treatment patterns, outcomes & burden, comparative options), \
  "intervention" should describe the real-world exposure/treatment pattern studied, \
  and "comparison" should be "Not applicable" unless the study explicitly compares \
  groups. For purely descriptive disease/epidemiology categories, use "Not applicable" \
  for any field the study design doesn't involve. Never leave a field blank - use \
  "Not applicable" or "Not specified".
- "summary": 2-3 sentence plain-language summary of the key finding
- "key_finding": one-sentence takeaway
- "study_type": "Systematic review", "Meta-analysis", "Randomized controlled trial", \
  "Cohort study", "Cross-sectional study", "Registry/claims analysis", "Case report", \
  "Narrative review", or "Not specified"
- "evidence_status": "Strong" (systematic review/meta-analysis/RCT), "Moderate" \
  (cohort/cross-sectional/registry/claims), "Limited" (case report/narrative review), \
  "Not specified" otherwise
- "confidence": "High", "Medium", or "Low" - your confidence the summary accurately \
  reflects the abstract

Use "Not specified" rather than guessing when information is absent.
"""


def _seconds_until_retry(error_message: str):
    match = re.search(r"try again in ([\d.]+)s", error_message)
    return float(match.group(1)) + 0.5 if match else None


def _is_rate_limit_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    return status == 429 or "429" in str(error) or "rate_limit_exceeded" in str(error)


def _parse_json_array(raw_text: str):
    """Handles reasoning-model output that wraps the JSON array in prose or
    an object instead of returning a bare array."""
    cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    parsed = json.loads(cleaned)
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                return value
        raise json.JSONDecodeError("No list found in JSON object", cleaned, 0)
    return parsed


def _extract_batch(abstracts: list, attempt: int = 0) -> list:
    user_content = json.dumps([
        {"pmid": a["pmid"], "category": a.get("category", "uncategorized"),
         "title": a["title"], "abstract": a["abstract"]}
        for a in abstracts
    ])
    by_key = {f"{a['pmid']}_{a.get('category', 'uncategorized')}": a for a in abstracts}

    def fallback(reason: str) -> list:
        return [{
            "pmid": a["pmid"], "category": a.get("category", "uncategorized"),
            "pico": {"population": "Not specified", "intervention": "Not specified",
                      "comparison": "Not specified", "outcome": "Not specified"},
            "summary": f"Could not extract this abstract ({reason}).",
            "key_finding": "Not specified", "study_type": "Not specified",
            "evidence_status": "Not specified", "confidence": "Low",
            "title": a.get("title", ""), "journal": a.get("journal", ""),
            "pub_date": a.get("pub_date", ""), "url": a.get("url", ""),
            "parse_ok": False,
        } for a in abstracts]

    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_content}],
            temperature=0.2, max_tokens=config.LLM_MAX_TOKENS,
        )
    except Exception as e:
        if _is_rate_limit_error(e) and attempt < config.LLM_RETRY_ATTEMPTS:
            wait = _seconds_until_retry(str(e)) or (config.LLM_RETRY_BASE_DELAY * (attempt + 1))
            time.sleep(wait)
            return _extract_batch(abstracts, attempt + 1)
        return fallback(f"API error: {e}")

    try:
        insights = _parse_json_array(response.choices[0].message.content)
    except json.JSONDecodeError:
        return fallback("malformed JSON response")

    merged = []
    for insight in insights:
        key = f"{insight.get('pmid')}_{insight.get('category')}"
        source = by_key.get(key, {})
        merged.append({**insight, "title": source.get("title", ""),
                        "journal": source.get("journal", ""), "pub_date": source.get("pub_date", ""),
                        "url": source.get("url", ""), "parse_ok": True})
    return merged


def extract_insights(abstracts: list) -> list:
    if not abstracts:
        return []
    all_insights = []
    batches = [abstracts[i:i + config.LLM_BATCH_SIZE] for i in range(0, len(abstracts), config.LLM_BATCH_SIZE)]
    for i, batch in enumerate(batches):
        if i > 0:
            time.sleep(1.5)
        all_insights.extend(_extract_batch(batch))
    return all_insights

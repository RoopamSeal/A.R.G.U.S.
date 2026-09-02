"""
Sends PubMed abstracts to an LLM (Groq / Llama 3.3) and gets back
structured insights as JSON - in PICO format, with evidence status,
confidence rating, and study-type metadata for each retrieved abstract.

Abstracts are sent in small batches rather than all at once. This matters:
sending 15+ abstracts in a single call risks a truncated or malformed
response, and since parsing was previously all-or-nothing, one bad response
used to wipe out every abstract's data - producing a misleadingly "empty"
evidence base instead of a clear error.
"""
import json
from groq import Groq
import config

client = Groq(api_key=config.GROQ_API_KEY)

SYSTEM_PROMPT = """You are a biomedical evidence analyst building a structured evidence \
repository. You will be given a list of PubMed abstracts, each tagged with a "category" \
(the section of a disease evidence framework it was retrieved for) and a "pmid".

For EACH abstract, extract a structured insight. Respond ONLY with a JSON array (no \
markdown, no preamble, no code fences). Each element must have exactly these fields:

- "pmid": echo back the pmid you were given, as a string
- "category": echo back the category you were given, exactly as given
- "pico": an object with "population", "intervention", "comparison", "outcome" - use \
  "Not applicable" for any field the study design doesn't involve (purely descriptive \
  epidemiology studies usually have no intervention/comparison)
- "summary": a 2-3 sentence plain-language summary of the key finding
- "key_finding": the single most important takeaway, in one sentence
- "study_type": your best read of the study design - one of "Systematic review", \
  "Meta-analysis", "Randomized controlled trial", "Cohort study", "Cross-sectional study", \
  "Case report", "Narrative review", "Not specified"
- "evidence_status": "Strong" if study_type is a systematic review, meta-analysis, or RCT; \
  "Moderate" if it's an observational/cohort/cross-sectional study; "Limited" if it's a \
  case report, case series, or narrative review; "Not specified" otherwise
- "confidence": "High", "Medium", or "Low" - how confident you are that this summary \
  accurately represents the abstract (lower this if the abstract is vague, truncated, \
  or ambiguous)

If information is not present in the abstract, use "Not specified" rather than guessing.
"""


def _extract_batch(abstracts: list) -> list:
    """Call the LLM for one small batch of abstracts. Returns merged insight
    dicts, each tagged with parse_ok=True/False so the caller (and the UI)
    can tell a real result apart from a parsing failure."""
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
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=config.LLM_MAX_TOKENS,
        )
    except Exception as e:
        return fallback(f"API error: {e}")

    raw_text = response.choices[0].message.content.strip()
    raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        insights = json.loads(raw_text)
    except json.JSONDecodeError:
        return fallback("malformed JSON response")

    merged = []
    for insight in insights:
        key = f"{insight.get('pmid')}_{insight.get('category')}"
        source = by_key.get(key, {})
        merged.append({
            **insight,
            "title": source.get("title", ""),
            "journal": source.get("journal", ""),
            "pub_date": source.get("pub_date", ""),
            "url": source.get("url", ""),
            "parse_ok": True,
        })
    return merged


def extract_insights(abstracts: list) -> list:
    """Send abstracts to the LLM in small batches and return structured
    insight dicts merged with source metadata. Each abstract in the input
    must already have a 'category' field (see
    pubmed_fetcher.fetch_module1_abstracts). A failure in one batch does not
    affect the others."""
    if not abstracts:
        return []

    all_insights = []
    for i in range(0, len(abstracts), config.LLM_BATCH_SIZE):
        batch = abstracts[i:i + config.LLM_BATCH_SIZE]
        all_insights.extend(_extract_batch(batch))
    return all_insights

"""
Sends PubMed abstracts to an LLM (Groq / Llama 3.3) and gets back
structured insights as JSON.
"""
import json
from groq import Groq
import config

client = Groq(api_key=config.GROQ_API_KEY)

SYSTEM_PROMPT = """You are a biomedical literature analyst. You will be given a list of \
PubMed abstracts. For EACH abstract, extract a structured insight.

Respond ONLY with a JSON array (no markdown, no preamble, no code fences). Each element must have:
- "pmid": the PubMed ID given to you (as a string)
- "summary": a 2-3 sentence plain-language summary of the key finding
- "population": the study population if stated, otherwise "Not specified"
- "key_finding": the single most important takeaway, in one sentence
- "confidence": one of "High", "Medium", "Low" - based on study size/design as described in the abstract

If information is not present in the abstract, use "Not specified" rather than guessing.
"""


def extract_insights(abstracts: list) -> list:
    """Send abstracts to the LLM and return structured insight dicts merged
    with source metadata (title, journal, url) for display."""
    if not abstracts:
        return []

    user_content = json.dumps([
        {"pmid": a["pmid"], "title": a["title"], "abstract": a["abstract"]}
        for a in abstracts
    ])

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    raw_text = response.choices[0].message.content.strip()
    raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        insights = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fall back gracefully instead of crashing the app on a malformed LLM response
        return [{
            "pmid": a["pmid"], "summary": "Could not parse LLM output for this abstract.",
            "population": "Not specified", "key_finding": "Not specified", "confidence": "Low",
        } for a in abstracts]

    by_pmid = {a["pmid"]: a for a in abstracts}
    merged = []
    for insight in insights:
        source = by_pmid.get(insight.get("pmid"), {})
        merged.append({
            **insight,
            "title": source.get("title", ""),
            "journal": source.get("journal", ""),
            "pub_date": source.get("pub_date", ""),
            "url": source.get("url", ""),
        })
    return merged

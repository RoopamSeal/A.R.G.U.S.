"""
v2: Generates a short executive-summary overview across every insight
retrieved for a topic, spanning all Module 1 sub-categories. This is what
answers "give me a summary of everything that's been fetched" instead of
making the user read every card individually.
"""
from groq import Groq
import config

client = Groq(api_key=config.GROQ_API_KEY)

OVERVIEW_SYSTEM_PROMPT = """You are a biomedical evidence analyst. You will be given a \
topic and a list of structured insights extracted from PubMed abstracts, grouped by \
category (disease definition, epidemiology, patient funnel, burden of disease, unmet need).

Write a concise executive summary (120-180 words, plain prose, no markdown headers) that:
- States what is well-established across the retrieved literature
- Flags where evidence is thin, limited to a single study, or conflicting
- Notes any of the five categories with little or no retrieved evidence
- Synthesizes across findings rather than listing every one individually

Respond with ONLY the summary paragraph, nothing else."""


def generate_overview(topic: str, insights: list) -> str:
    """Summarize the full set of retrieved insights into one short paragraph."""
    if not insights:
        return "No insights were retrieved for this topic."

    # Only send the fields the summary actually needs - keeps the prompt small and cheap
    condensed = [
        {
            "category": i.get("category"),
            "key_finding": i.get("key_finding"),
            "evidence_status": i.get("evidence_status"),
        }
        for i in insights
    ]

    user_content = f"Topic: {topic}\n\nInsights:\n{condensed}"

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": OVERVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

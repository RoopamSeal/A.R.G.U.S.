"""Generate a cautious executive synthesis across a module's retrieved records."""
from groq import Groq
import config

PROMPT = """You are an evidence analyst supporting early Market Access decisions.
Write a concise 120-180 word overview in plain prose. State what is established in the retrieved records,
where evidence is thin, conflicting or indirect, and which categories had little/no evidence.
Separate study findings from interpretation. Do not invent regulatory status, HTA outcomes, prices,
market sizes or causal conclusions. Explicitly state that this is a synthesis of the supplied records,
not a systematic review or a verified market assessment."""

def generate_overview(topic, insights, module_label="evidence module"):
    if not insights:
        return "No insights were retrieved for this topic."
    condensed=[{"category":i.get("category"),"key_finding":i.get("key_finding"),
                "evidence_status":i.get("evidence_status"),"study_type":i.get("study_type"),
                "parse_ok":i.get("parse_ok")} for i in insights]
    response=Groq(api_key=config.GROQ_API_KEY).chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role":"system","content":PROMPT},
                  {"role":"user","content":f"Module: {module_label}\nTopic: {topic}\nRecords: {condensed}"}],
        temperature=0.2,max_tokens=500)
    return response.choices[0].message.content.strip()

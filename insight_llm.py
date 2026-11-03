"""Extract traceable structured insights from PubMed abstracts using Groq."""
import json
import re
import time
from groq import Groq
import config

SYSTEM_PROMPT = """You are a cautious biomedical evidence analyst supporting early Market Access strategy.
For EACH supplied PubMed abstract, return one JSON object in a JSON array. Return ONLY JSON.
Fields: pmid (string), category (exact input category), pico {population, intervention, comparison, outcome},
summary (2-3 factual sentences), key_finding (one sentence), study_type, evidence_status
(Strong, Moderate, Limited, Not specified), confidence (High, Moderate, Low).
For descriptive, regulatory, pricing or observational material, use Not applicable where PICO does not fit.
Never infer results, prices, approvals, reimbursement or causality not stated in the abstract.
Use Not specified for missing facts. Confidence rates extraction fidelity, not certainty that a claim is true.
Do not upgrade evidence strength solely because an abstract calls a result significant."""

def _client():
    return Groq(api_key=config.GROQ_API_KEY)

def _parse_json_array(raw):
    cleaned = (raw or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("Model response did not contain a JSON array")
    parsed = json.loads(cleaned[start:end+1])
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array")
    return parsed

def _fallback(a, reason):
    return {"pmid":str(a["pmid"]), "category":a.get("category","uncategorized"),
            "pico":{"population":"Not specified","intervention":"Not specified",
                    "comparison":"Not specified","outcome":"Not specified"},
            "summary":f"Extraction unavailable ({reason}).", "key_finding":"Not specified",
            "study_type":"Not specified","evidence_status":"Not specified","confidence":"Low",
            "title":a.get("title",""),"journal":a.get("journal",""),"pub_date":a.get("pub_date",""),
            "url":a.get("url",""),"parse_ok":False}

def _extract_batch(abstracts, attempt=0):
    user_content=json.dumps([{"pmid":str(a["pmid"]),"category":a.get("category","uncategorized"),
                              "title":a.get("title",""),"abstract":a.get("abstract","")} for a in abstracts])
    try:
        response=_client().chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_content}],
            temperature=0.1,max_tokens=config.LLM_MAX_TOKENS)
        parsed=_parse_json_array(response.choices[0].message.content)
        by_id={(str(a["pmid"]),a.get("category","uncategorized")):a for a in abstracts}
        results=[]
        for item in parsed:
            if not isinstance(item,dict):
                continue
            source=by_id.get((str(item.get("pmid","")),item.get("category","uncategorized")))
            if source:
                results.append({**_fallback(source,"missing fields"),**item,
                    "title":source.get("title",""),"journal":source.get("journal",""),
                    "pub_date":source.get("pub_date",""),"url":source.get("url",""),"parse_ok":True})
        # Guarantee one visible row per retrieved source; do not silently drop omissions.
        found={(str(x.get("pmid")),x.get("category")) for x in results}
        for a in abstracts:
            key=(str(a["pmid"]),a.get("category","uncategorized"))
            if key not in found:
                results.append(_fallback(a,"not returned by model"))
        return results
    except Exception as exc:
        message=str(exc)
        if ("429" in message or "rate_limit" in message) and attempt < config.LLM_RETRY_ATTEMPTS:
            match=re.search(r"try again in ([\d.]+)s",message)
            time.sleep(float(match.group(1))+0.5 if match else config.LLM_RETRY_BASE_DELAY*(attempt+1))
            return _extract_batch(abstracts,attempt+1)
        return [_fallback(a,str(exc)[:180]) for a in abstracts]

def extract_insights(abstracts):
    if not abstracts:
        return []
    output=[]
    for i in range(0,len(abstracts),config.LLM_BATCH_SIZE):
        if i:
            time.sleep(1)
        output.extend(_extract_batch(abstracts[i:i+config.LLM_BATCH_SIZE]))
    return output

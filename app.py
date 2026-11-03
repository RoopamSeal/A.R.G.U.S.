"""
A.R.G.U.S. — AI Rapid Generation of Useful Strategy
Decision-oriented early Market Access workspace.
"""
import json
from datetime import date, datetime, timezone
import streamlit as st
import config
import pubmed_fetcher
import insight_llm
import summary_llm
import cache

st.set_page_config(page_title="A.R.G.U.S. | Market Access Intelligence", page_icon="🧭", layout="wide")

try:
    config.validate_config()
except EnvironmentError as exc:
    st.error(str(exc))
    st.info("Configure GROQ_API_KEY and ENTREZ_EMAIL in your environment, then reload A.R.G.U.S.")
    st.stop()

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
.argus-kicker {letter-spacing:.12em;text-transform:uppercase;font-size:.72rem;font-weight:700;color:#64748b}
.argus-card {border:1px solid rgba(128,128,128,.28);border-radius:12px;padding:15px 17px;margin:.35rem 0}
.argus-muted {color:#64748b;font-size:.88rem}
</style>
""", unsafe_allow_html=True)

if "argus_workspace" not in st.session_state:
    st.session_state.argus_workspace = {
        "asset": "", "indication": "", "stage": "Phase II", "cutoff": str(date.today()),
        "market": "Global / priority markets", "thesis": "", "attractiveness": "Not assessed",
        "maturity": "Not assessed", "value_drivers": "", "risks": "", "tap": "",
        "recommendation": "Reassess", "conditions": "", "actions": "", "gaps": [],
        "scenarios": {"Floor": "", "Target": "", "Ceiling": ""},
        "manual_evidence": [], "module_results": {}
    }
ws = st.session_state.argus_workspace

def section(title, subtitle=None):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)

def render_insight(item):
    with st.container(border=True):
        st.markdown(f"**{item.get('title', 'Untitled study')}**")
        if item.get("parse_ok") is False:
            st.warning(item.get("summary", "Extraction unavailable."))
        else:
            st.write(item.get("summary", ""))
        pico = item.get("pico", {}) or {}
        with st.expander("PICO, finding & evidence assessment", expanded=False):
            cols = st.columns(4)
            for col, label, key in zip(cols, ["Population", "Intervention", "Comparator", "Outcome"],
                                       ["population", "intervention", "comparison", "outcome"]):
                col.markdown(f"**{label}**")
                col.write(pico.get(key, "Not specified"))
            st.markdown(f"**Key finding:** {item.get('key_finding', 'Not specified')}")
            st.write(f"Study type: {item.get('study_type', 'Not specified')} · Evidence strength: {item.get('evidence_status', 'Not specified')} · Extraction confidence: {item.get('confidence', 'Low')}")
        st.caption(f"{item.get('journal', 'Journal not supplied')} · {item.get('pub_date', 'Date unknown')} · PMID {item.get('pmid', 'unknown')}")
        if item.get("url"):
            st.markdown(f"[Open source record ↗]({item['url']})")

def run_module(module_key, meta):
    number = meta["number"]
    cats = meta["categories"]
    st.title(f"Module {number} | {meta['label']}")
    st.caption(meta["caption"])
    st.info("Literature retrieval uses PubMed. Regulatory, HTA, reimbursement and price fields must be validated against the relevant official agency or local source; PubMed results alone are not an authoritative source for those decisions.")
    topic = st.text_input("Asset, disease or indication", value=ws["indication"], key=f"{module_key}_topic",
                          placeholder="Enter a disease, indication, asset or relevant search topic")
    left, right = st.columns([1, 1])
    with left:
        n = st.slider("Abstracts per evidence category", 1, 8, config.ABSTRACTS_PER_CATEGORY, key=f"{module_key}_n")
    with right:
        fresh = st.checkbox("Force fresh retrieval (bypass cache)", value=False, key=f"{module_key}_fresh")
    if st.button(f"Generate Module {number} evidence", type="primary", key=f"{module_key}_run", disabled=not topic.strip()):
        cache_key = f"v2::{module_key}::{topic.strip().lower()}::{n}"
        cached = None if fresh else cache.get(cache_key)
        if isinstance(cached, dict) and "categories" in cached:
            result = cached
            st.success("Loaded previously generated results from cache.")
        else:
            search_categories = {
                key: {"label": key.replace("_", " "), "search_terms": terms}
                for key, terms in cats.items()
            }
            with st.spinner(f"Searching PubMed for {len(cats)} evidence categories…"):
                abstracts = pubmed_fetcher.fetch_categorized_abstracts(topic, search_categories, n)
            if not abstracts:
                st.warning("No abstracts with usable records were returned. Try a broader search term.")
                st.stop()
            with st.spinner(f"Extracting structured insights from {len(abstracts)} abstracts…"):
                insights = insight_llm.extract_insights(abstracts)
            with st.spinner("Preparing the evidence overview…"):
                overview = summary_llm.generate_overview(topic, insights, meta["label"])
            grouped = {key: [] for key in cats}
            for item in insights:
                grouped.setdefault(item.get("category", "uncategorized"), []).append(item)
            result = {"topic": topic, "module": module_key, "categories": grouped,
                      "overview": overview, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                      "abstract_count": len(abstracts)}
            cache.set(cache_key, result)
        ws["module_results"][module_key] = result
        st.rerun()

    result = ws["module_results"].get(module_key)
    if result:
        st.success(f"{sum(len(v) for v in result['categories'].values())} structured records · retrieved {result.get('retrieved_at','')}")
        section("Evidence overview")
        st.write(result.get("overview", ""))
        for key, terms in cats.items():
            items = result["categories"].get(key, [])
            with st.expander(f"{key.replace('_',' ').title()} · {len(items)} records", expanded=bool(items)):
                if not items:
                    st.caption("No records retrieved in this category.")
                for item in items:
                    render_insight(item)
        st.download_button("Export module evidence (JSON)", json.dumps(result, indent=2),
                           file_name=f"argus_{module_key}_evidence.json", mime="application/json",
                           key=f"{module_key}_export")
    else:
        st.caption("Run a search to populate this evidence module. Generated results will remain in this workspace session and can be exported.")

st.title("🧭 A.R.G.U.S.")
st.markdown("**AI Rapid Generation of Useful Strategy**")
st.caption("Early Market Access intelligence · Phase I / Phase II / Business Development · Evidence-backed decision support")

main_tabs = st.tabs(["Executive dashboard", "Evidence modules", "Strategic synthesis", "Evidence gaps & governance", "Workspace export"])

with main_tabs[0]:
    section("Executive decision dashboard", "Landing page for investment, development and early Market Access decisions.")
    c1,c2,c3,c4 = st.columns(4)
    ws["asset"] = c1.text_input("Asset / molecule", value=ws["asset"])
    ws["indication"] = c2.text_input("Indication / disease", value=ws["indication"])
    ws["stage"] = c3.selectbox("Development stage", ["Discovery", "Preclinical", "Phase I", "Phase II", "Phase III", "Pre-registration", "Marketed", "Business Development"],
                               index=["Discovery","Preclinical","Phase I","Phase II","Phase III","Pre-registration","Marketed","Business Development"].index(ws["stage"]) if ws["stage"] in ["Discovery","Preclinical","Phase I","Phase II","Phase III","Pre-registration","Marketed","Business Development"] else 3)
    ws["cutoff"] = c4.date_input("Evidence cut-off", value=date.fromisoformat(ws["cutoff"])).isoformat()
    ws["market"] = st.text_input("Priority markets / geography", value=ws["market"])
    st.divider()
    a,b,c = st.columns(3)
    ws["attractiveness"] = a.selectbox("Access attractiveness", ["Not assessed","High","Moderate","Low","Conditional"], index=["Not assessed","High","Moderate","Low","Conditional"].index(ws["attractiveness"]))
    ws["maturity"] = b.selectbox("Evidence maturity", ["Not assessed","High","Moderate","Low","Early / immature"], index=["Not assessed","High","Moderate","Low","Early / immature"].index(ws["maturity"]))
    ws["recommendation"] = c.selectbox("Recommendation", ["Proceed","Proceed with conditions","Reassess"], index=["Proceed","Proceed with conditions","Reassess"].index(ws["recommendation"]))
    ws["thesis"] = st.text_area("Market Access thesis", value=ws["thesis"], placeholder="Strongest population and positioning; value drivers to demonstrate; likely comparator; access opportunity; principal condition or risk.")
    c1,c2 = st.columns(2)
    ws["value_drivers"] = c1.text_area("Potential value drivers (3–5)", value=ws["value_drivers"], placeholder="Clinical outcomes, disease modification, safety, convenience, HRQoL, HCRU or system benefit.")
    ws["risks"] = c2.text_area("Major access risks", value=ws["risks"], placeholder="Comparator, endpoint, generalisability, affordability, future competition, operational burden and country-specific evidence.")
    ws["tap"] = st.text_area("Minimum Target Access Profile (TAP)", value=ws["tap"], placeholder="Payer-relevant population, comparator, efficacy, safety, endpoints, follow-up, administration, HRQoL/utility and economic evidence.")
    st.markdown("**Price & access scenarios**")
    p1,p2,p3 = st.columns(3)
    for col, label in zip([p1,p2,p3], ["Floor","Target","Ceiling"]):
        ws["scenarios"][label] = col.text_input(f"{label} scenario", value=ws["scenarios"].get(label,""), placeholder="Price, population, assumptions, restrictions")
    ws["conditions"] = st.text_area("Top conditions for success", value=ws["conditions"])
    ws["actions"] = st.text_area("Five immediate actions / next decision point", value=ws["actions"])
    st.divider()
    st.markdown("**Module completion**")
    completed = sum(1 for k in config.MODULES if k in ws["module_results"])
    st.progress(completed / len(config.MODULES), text=f"{completed} of 7 evidence modules generated in this session")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Evidence modules", f"{completed}/7")
    metric_cols[1].metric("Evidence records", sum(sum(len(v) for v in r["categories"].values()) for r in ws["module_results"].values()))
    metric_cols[2].metric("Open evidence gaps", len(ws["gaps"]))
    metric_cols[3].metric("Manual evidence entries", len(ws["manual_evidence"]))

with main_tabs[1]:
    section("Seven evidence modules", "Each module follows the evidence → implication → gap → action → confidence logic.")
    chosen = st.selectbox("Select module", list(config.MODULES.keys()), format_func=lambda k: f"{config.MODULES[k]['number']}. {config.MODULES[k]['label']}")
    run_module(chosen, config.MODULES[chosen])

with main_tabs[2]:
    section("Strategic synthesis", "Translate evidence into an integrated Market Access and development recommendation.")
    synth_fields = [
        ("opportunity", "Opportunity & north star", "Priority population, future treatment setting, pathway role and conditions for success."),
        ("payer_story", "Early payer value story", "Prioritised value hypotheses and evidence required to substantiate each claim."),
        ("pico", "Initial PICO scenarios", "Population, intervention, comparator and outcomes; distinguish prediction from confirmed HTA PICO."),
        ("access", "Price & access strategy", "Comparator/price anchors, affordability, population breadth versus price and likely restrictions."),
        ("generation", "Evidence generation plan", "Clinical programme, Phase II/III data, payer research, HTA advice, ITC, HE model and RWE."),
        ("bd", "Portfolio / Business Development implications", "Investment attractiveness, deal conditions, indication sequencing and value inflection points."),
    ]
    if "synthesis" not in ws: ws["synthesis"] = {}
    for key,label,hint in synth_fields:
        ws["synthesis"][key] = st.text_area(label, value=ws["synthesis"].get(key,""), placeholder=hint, key=f"synthesis_{key}")
    st.markdown("**Final recommendation**")
    st.write(f"Current dashboard recommendation: **{ws['recommendation']}**")
    ws["synthesis"]["conditions"] = st.text_area("Top three conditions that would change the recommendation", value=ws["synthesis"].get("conditions",""))
    ws["synthesis"]["next_gate"] = st.text_input("Next development / deal decision point", value=ws["synthesis"].get("next_gate",""))

with main_tabs[3]:
    section("Prioritised evidence-gap register", "Rank gaps by their impact on a decision, not simply by data availability.")
    with st.form("gap_form", clear_on_submit=True):
        g1,g2 = st.columns(2)
        gap_title = g1.text_input("Evidence gap / question")
        gap_priority = g2.selectbox("Priority", ["Critical","High","Medium","Monitor"])
        gap_decision = st.text_input("Decision affected")
        gap_action = st.text_area("Recommended evidence response / action")
        g3,g4 = st.columns(2)
        gap_owner = g3.text_input("Owner function")
        gap_due = g4.text_input("Deadline / decision date")
        gap_conf = st.selectbox("Confidence in gap assessment", config.CONFIDENCE_LEVELS)
        submitted = st.form_submit_button("Add evidence gap")
        if submitted and gap_title.strip():
            ws["gaps"].append({"gap":gap_title,"priority":gap_priority,"decision":gap_decision,
                               "action":gap_action,"owner":gap_owner,"deadline":gap_due,"confidence":gap_conf})
    if ws["gaps"]:
        for i,gap in enumerate(ws["gaps"]):
            with st.container(border=True):
                x,y = st.columns([5,1])
                x.markdown(f"**{gap['priority']} · {gap['gap']}**")
                x.caption(f"Decision: {gap['decision']} · Owner: {gap['owner']} · Deadline: {gap['deadline']} · Confidence: {gap['confidence']}")
                x.write(gap["action"])
                if y.button("Remove", key=f"remove_gap_{i}"):
                    ws["gaps"].pop(i)
                    st.rerun()
    else:
        st.caption("No evidence gaps recorded yet.")
    st.divider()
    section("Evidence quality, confidence & governance")
    st.markdown("""
- **Verified evidence:** direct source-supported fact.
- **AI synthesis / interpretation:** generated interpretation; requires expert review.
- **Forecast assumption:** scenario, projection or unvalidated assumption.
- **Evidence gap:** information is absent, insufficient or unresolved.

Confidence: **High** = authoritative and reproducible; **Moderate** = credible but indirect, heterogeneous or incomplete; **Low** = estimate, projection, limited public data or unvalidated assumption.
""")
    with st.form("evidence_form", clear_on_submit=True):
        e1,e2 = st.columns(2)
        ev_claim = e1.text_input("Claim / extracted evidence")
        ev_status = e2.selectbox("Evidence status", config.EVIDENCE_STATUSES)
        e3,e4 = st.columns(2)
        ev_source = e3.text_input("Source title / URL / reference")
        ev_date = e4.text_input("Publication / decision date")
        e5,e6 = st.columns(2)
        ev_geo = e5.text_input("Geography / population")
        ev_conf = e6.selectbox("Confidence", config.CONFIDENCE_LEVELS)
        ev_uncertainty = st.text_area("Uncertainty, limitations or contradiction")
        if st.form_submit_button("Add evidence record") and ev_claim.strip():
            ws["manual_evidence"].append({"claim":ev_claim,"status":ev_status,"source":ev_source,
                "date":ev_date,"geography_population":ev_geo,"confidence":ev_conf,"uncertainty":ev_uncertainty})
    if ws["manual_evidence"]:
        st.dataframe(ws["manual_evidence"], use_container_width=True, hide_index=True)
    st.warning("Before decision use: HEOR / Market Access review strategic interpretation; technical review modelling, ITC and RWE; pricing validation for net-price assumptions; local country validation before submission use.")

with main_tabs[4]:
    section("Evidence annex & audit trail", "Export the current workspace, module results, gap register and evidence log.")
    st.markdown("The export contains only information entered or generated in this session. It does not imply independent verification of source claims.")
    payload = {
        "product": "A.R.G.U.S.", "schema_version": "2.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "project": ws,
        "module_definitions": {k: {"number":v["number"],"label":v["label"],"categories":v["categories"]} for k,v in config.MODULES.items()},
        "governance": {"evidence_statuses":config.EVIDENCE_STATUSES,"confidence_levels":config.CONFIDENCE_LEVELS,
                       "evidence_cutoff":ws.get("cutoff"),
                       "note":"Retain source links, publication/decision dates, geography, population, comparator, outcome/timepoint, evidence type, uncertainty and last update."}
    }
    st.download_button("Download complete A.R.G.U.S. workspace (JSON)", json.dumps(payload, indent=2, ensure_ascii=False),
                       file_name="argus_market_access_workspace.json", mime="application/json", type="primary")
    st.download_button("Download evidence-gap register (CSV)", 
                       "priority,gap,decision,action,owner,deadline,confidence\n" + "\n".join(
                           ",".join('"'+str(g.get(k,"")).replace('"','""')+'"' for k in ["priority","gap","decision","action","owner","deadline","confidence"])
                           for g in ws["gaps"]),
                       file_name="argus_evidence_gap_register.csv", mime="text/csv")
    st.caption("Source hierarchy recommended by the framework: regulators, HTA agencies, trial registries, guidelines, peer-reviewed literature and official pricing/formulary sources. Preserve contradictory estimates and document search strategy and evidence cut-off.")

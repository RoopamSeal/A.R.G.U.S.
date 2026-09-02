"""
PubMed Insight Generator - v2
Now structured around the evidence framework: a tab per module (only
Module 1 is built so far), five sub-category cards within it, and each
retrieved insight shown with PICO fields, evidence status, confidence
rating, source metadata, and a link back to PubMed - plus one overall
summary of everything that was fetched.
"""
import streamlit as st
import config
import pubmed_fetcher
import insight_llm
import summary_llm
import cache

st.set_page_config(page_title="Evidence Insight Repository", page_icon="🧬", layout="wide")

# Fail fast with a clear message if API keys aren't set, instead of a confusing traceback later
try:
    config.validate_config()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

CONFIDENCE_ICON = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
EVIDENCE_ICON = {"Strong": "🟢", "Moderate": "🟡", "Limited": "🟠", "Not specified": "⚪"}


def render_insight_card(item: dict):
    with st.container(border=True):
        st.markdown(f"**{item.get('title', 'Untitled')}**")
        st.write(item.get("summary", ""))

        pico = item.get("pico", {})
        p_cols = st.columns(4)
        p_cols[0].markdown(f"**Population**\n\n{pico.get('population', 'Not specified')}")
        p_cols[1].markdown(f"**Intervention**\n\n{pico.get('intervention', 'Not specified')}")
        p_cols[2].markdown(f"**Comparison**\n\n{pico.get('comparison', 'Not specified')}")
        p_cols[3].markdown(f"**Outcome**\n\n{pico.get('outcome', 'Not specified')}")

        st.markdown(f"**Key finding:** {item.get('key_finding', 'Not specified')}")

        m_cols = st.columns([2, 2, 2, 3])
        ev = item.get("evidence_status", "Not specified")
        conf = item.get("confidence", "Unknown")
        m_cols[0].markdown(f"{EVIDENCE_ICON.get(ev, '⚪')} Evidence: **{ev}**")
        m_cols[1].markdown(f"{CONFIDENCE_ICON.get(conf, '⚪')} Confidence: **{conf}**")
        m_cols[2].markdown(f"🧪 {item.get('study_type', 'Not specified')}")
        m_cols[3].markdown(f"📅 {item.get('pub_date', 'Unknown')} · {item.get('journal', '')}")

        st.markdown(f"[View source on PubMed ↗]({item.get('url', '#')})")


tabs = st.tabs(config.ALL_MODULES)

# --- Module 1: Disease & burden (the only module built so far) ---
with tabs[0]:
    st.title("Module 1 | Disease, burden, treatment landscape and unmet need")
    st.caption("Establishes the decision context and quantifies why an improved intervention is needed.")

    topic = st.text_input(
        "Enter a disease or condition",
        placeholder="e.g. type 2 diabetes",
        key="module1_topic",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        abstracts_per_category = st.slider(
            "Abstracts per sub-category", min_value=2, max_value=6,
            value=config.ABSTRACTS_PER_CATEGORY,
        )
    with col2:
        use_cache = st.checkbox("Use cache if available", value=True)

    if st.button("Generate Module 1 evidence", type="primary") and topic:
        cached = cache.get(topic) if use_cache else None

        if cached:
            st.info("Loaded from cache — uncheck 'Use cache' to force a fresh search.")
            categories_data = cached["categories"]
            overview = cached["overview"]
        else:
            with st.spinner("Searching PubMed across 5 sub-categories..."):
                abstracts = pubmed_fetcher.fetch_module1_abstracts(topic, abstracts_per_category)

            if not abstracts:
                st.warning("No PubMed abstracts found for this topic. Try a broader term.")
                st.stop()

            with st.spinner(f"Extracting structured insights from {len(abstracts)} abstracts..."):
                insights = insight_llm.extract_insights(abstracts)

            with st.spinner("Writing the evidence overview..."):
                overview = summary_llm.generate_overview(topic, insights)

            categories_data = {key: [] for key in config.MODULE_1_CATEGORIES}
            for item in insights:
                categories_data.setdefault(item.get("category", "uncategorized"), []).append(item)

            cache.set(topic, {"categories": categories_data, "overview": overview})

        total_insights = sum(len(v) for v in categories_data.values())
        st.success(f"Retrieved {total_insights} insight(s) for '{topic}' across 5 sub-categories")

        st.subheader("Evidence overview")
        st.info(overview)

        for category_key, meta in config.MODULE_1_CATEGORIES.items():
            items = categories_data.get(category_key, [])
            bar_color = meta["color"]
            bar_label = meta["label"].upper()
            st.markdown(
                f"<div style='background:{bar_color};color:white;padding:8px 14px;"
                f"border-radius:6px;font-weight:600;margin-top:18px;'>"
                f"{bar_label} ({len(items)})</div>",
                unsafe_allow_html=True,
            )
            if not items:
                st.caption("No PubMed evidence retrieved for this category.")
            else:
                for item in items:
                    render_insight_card(item)

        st.divider()
        st.caption(
            "Treatment pathway & standard of care, strategic interpretation, output "
            "visuals, and decision-relevant gaps are planned for a future version."
        )

    elif not topic:
        st.caption("👆 Enter a disease or condition above and click **Generate Module 1 evidence** to get started.")

# --- Remaining 6 modules: not built yet ---
for tab, module_name in zip(tabs[1:], config.ALL_MODULES[1:]):
    with tab:
        st.title(f"Module | {module_name}")
        st.caption("🚧 Coming in a future version.")

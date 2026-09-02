"""
Evidence Insight Repository - v3
Scoped to just two modules: Module 1 (Disease & burden) and Module 7
(Real-world evidence). Both modules share the exact same pipeline and
rendering logic (run_module below) - only their category definitions in
config.py differ. Each retrieved insight shows PICO fields, evidence
status, confidence rating, source metadata, and a link back to PubMed,
plus one overall summary of everything fetched for that module.
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


def run_module(module_number: int, module_title: str, module_caption: str,
                categories: dict, cache_prefix: str, widget_key: str,
                future_sections_note: str):
    """Renders one full module: topic input, generate button, overview
    summary, and one card group per sub-category. Module 1 and Module 7
    both call this with their own category set and cache namespace."""
    st.title(f"Module {module_number} | {module_title}")
    st.caption(module_caption)

    topic = st.text_input(
        "Enter a disease or condition",
        placeholder="e.g. type 2 diabetes",
        key=f"{widget_key}_topic",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        abstracts_per_category = st.slider(
            "Abstracts per sub-category", min_value=2, max_value=6,
            value=config.ABSTRACTS_PER_CATEGORY, key=f"{widget_key}_slider",
        )
    with col2:
        use_cache = st.checkbox("Use cache if available", value=True, key=f"{widget_key}_cache_toggle")

    generate_clicked = st.button(
        f"Generate Module {module_number} evidence", type="primary", key=f"{widget_key}_btn"
    )

    if generate_clicked and topic:
        # Prefix the cache key by module so the same disease name searched in
        # both modules doesn't collide - Module 1 and Module 7 cache separately.
        cache_key = f"{cache_prefix}::{topic.strip().lower()}"
        cached = cache.get(cache_key) if use_cache else None

        is_valid_cache = isinstance(cached, dict) and "categories" in cached and "overview" in cached

        if is_valid_cache:
            st.info("Loaded from cache — uncheck 'Use cache' to force a fresh search.")
            categories_data = cached["categories"]
            overview = cached["overview"]
        else:
            with st.spinner(f"Searching PubMed across {len(categories)} sub-categories..."):
                abstracts = pubmed_fetcher.fetch_categorized_abstracts(topic, categories, abstracts_per_category)

            if not abstracts:
                st.warning("No PubMed abstracts found for this topic. Try a broader term.")
                st.stop()

            with st.spinner(f"Extracting structured insights from {len(abstracts)} abstracts..."):
                insights = insight_llm.extract_insights(abstracts)

            with st.spinner("Writing the evidence overview..."):
                overview = summary_llm.generate_overview(topic, insights)

            categories_data = {key: [] for key in categories}
            for item in insights:
                categories_data.setdefault(item.get("category", "uncategorized"), []).append(item)

            cache.set(cache_key, {"categories": categories_data, "overview": overview})

        total_insights = sum(len(v) for v in categories_data.values())
        st.success(f"Retrieved {total_insights} insight(s) for '{topic}' across {len(categories)} sub-categories")

        st.subheader("Evidence overview")
        st.info(overview)

        for category_key, meta in categories.items():
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
        st.caption(future_sections_note)

    elif not topic:
        st.caption(f"👆 Enter a disease or condition above and click **Generate Module {module_number} evidence** to get started.")


tab1, tab7 = st.tabs(["Disease & burden", "Real-world evidence"])

with tab1:
    run_module(
        module_number=1,
        module_title="Disease, burden, treatment landscape and unmet need",
        module_caption="Establishes the decision context and quantifies why an improved intervention is needed.",
        categories=config.MODULE_1_CATEGORIES,
        cache_prefix="module1",
        widget_key="m1",
        future_sections_note=(
            "Treatment pathway & standard of care, strategic interpretation, output "
            "visuals, and decision-relevant gaps are planned for a future version."
        ),
    )

with tab7:
    run_module(
        module_number=7,
        module_title="Real-world evidence",
        module_caption="Connect every proposed RWE activity to a specific Market Access, trial, or economic decision.",
        categories=config.MODULE_7_CATEGORIES,
        cache_prefix="module7",
        widget_key="m7",
        future_sections_note=(
            "RWE readiness map, gap-to-study matrix, HE model inputs, and method & "
            "interpretation cautions are planned for a future version."
        ),
    )

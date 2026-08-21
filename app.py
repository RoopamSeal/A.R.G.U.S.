"""
PubMed Insight Generator - v1
Streamlit UI tying the pipeline together:
query -> PubMed fetch -> LLM insight extraction -> insight cards
"""
import streamlit as st
import config
import pubmed_fetcher
import insight_llm
import cache

st.set_page_config(page_title="A.R.G.U.S.", page_icon="🧬", layout="centered")

st.title("A.R.G.U.S. Insights")
st.caption("Retrieves literature from PubMed and get structured insights.")

# Fail fast with a clear message if API keys aren't set, instead of a confusing traceback later
try:
    config.validate_config()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

query = st.text_input(
    "Enter a disease, drug, or topic",
    placeholder="e.g. metformin cardiovascular outcomes",
)

col1, col2 = st.columns([1, 1])
with col1:
    max_results = st.slider("Number of abstracts to analyze", min_value=3, max_value=15, value=8)
with col2:
    use_cache = st.checkbox("Use cache if available", value=True)

if st.button("Generate insights", type="primary") and query:
    cached = cache.get(query) if use_cache else None

    if cached:
        st.info("Loaded from cache — uncheck 'Use cache' to force a fresh search.")
        insights = cached
    else:
        with st.spinner("Searching PubMed..."):
            abstracts = pubmed_fetcher.search_and_fetch(query, max_results)

        if not abstracts:
            st.warning("No PubMed abstracts found for this query. Try a broader term.")
            st.stop()

        with st.spinner(f"Extracting insights from {len(abstracts)} abstracts with the LLM..."):
            insights = insight_llm.extract_insights(abstracts)

        cache.set(query, insights)

    st.success(f"Found {len(insights)} insight(s) for '{query}'")

    for item in insights:
        with st.container(border=True):
            st.markdown(f"**{item.get('title', 'Untitled')}**")
            st.write(item.get("summary", ""))
            st.markdown(f"**Key finding:** {item.get('key_finding', 'Not specified')}")
            st.markdown(f"**Population:** {item.get('population', 'Not specified')}")

            meta_cols = st.columns([2, 2, 2])
            meta_cols[2].markdown(f"[View on PubMed ↗]({item.get('url', '#')})")

elif not query:
    st.caption("👆 Enter a topic above and click **Generate insights** to get started.")

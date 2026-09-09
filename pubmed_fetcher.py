"""
Fetches abstracts from PubMed using NCBI's Entrez API (via Biopython).
"""
from Bio import Entrez
import config

Entrez.email = config.ENTREZ_EMAIL
if config.ENTREZ_API_KEY:
    Entrez.api_key = config.ENTREZ_API_KEY


def search_pubmed(query: str, max_results: int = None) -> list:
    max_results = max_results or config.MAX_ABSTRACTS_PER_QUERY
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])


def fetch_abstracts(pmids: list) -> list:
    if not pmids:
        return []
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    articles = []
    for article in records.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            article_data = medline["Article"]
            pmid = str(medline["PMID"])
            title = str(article_data.get("ArticleTitle", ""))
            abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(part) for part in abstract_parts) if abstract_parts else ""
            journal = str(article_data.get("Journal", {}).get("Title", "Unknown journal"))
            pub_date_info = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            pub_date = pub_date_info.get("Year", pub_date_info.get("MedlineDate", "Unknown date"))
            if abstract:
                articles.append({
                    "pmid": pmid, "title": title, "abstract": abstract,
                    "journal": journal, "pub_date": str(pub_date),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
        except (KeyError, IndexError):
            continue
    return articles


def search_and_fetch(query: str, max_results: int = None) -> list:
    pmids = search_pubmed(query, max_results)
    return fetch_abstracts(pmids)


def fetch_categorized_abstracts(topic: str, categories: dict, abstracts_per_category: int = None) -> list:
    """One search per sub-category, terms OR'd together for recall. Falls
    back to a broader query if a category returns nothing."""
    abstracts_per_category = abstracts_per_category or config.ABSTRACTS_PER_CATEGORY
    seen_pmids = set()
    all_abstracts = []

    for category_key, meta in categories.items():
        terms_clause = " OR ".join(meta["search_terms"])
        query = f"{topic} AND ({terms_clause})"
        results = search_and_fetch(query, abstracts_per_category)
        if not results:
            results = search_and_fetch(f"{topic} {meta['label']}", abstracts_per_category)
        for article in results:
            if article["pmid"] in seen_pmids:
                continue
            seen_pmids.add(article["pmid"])
            article["category"] = category_key
            all_abstracts.append(article)

    return all_abstracts

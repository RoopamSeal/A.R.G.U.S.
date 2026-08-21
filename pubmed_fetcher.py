"""
Fetches abstracts from PubMed using NCBI's Entrez API (via Biopython).
"""
from Bio import Entrez
import config

Entrez.email = config.ENTREZ_EMAIL
if config.ENTREZ_API_KEY:
    Entrez.api_key = config.ENTREZ_API_KEY


def search_pubmed(query: str, max_results: int = None) -> list:
    """Search PubMed and return a list of matching PubMed IDs (PMIDs)."""
    max_results = max_results or config.MAX_ABSTRACTS_PER_QUERY
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])


def fetch_abstracts(pmids: list) -> list:
    """Given a list of PMIDs, fetch title, abstract, journal, and pub date for each."""
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

            if abstract:  # skip records with no abstract text - nothing for the LLM to summarize
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "pub_date": str(pub_date),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
        except (KeyError, IndexError):
            continue  # skip malformed records rather than crashing the whole batch

    return articles


def search_and_fetch(query: str, max_results: int = None) -> list:
    """Convenience wrapper: search + fetch in one call."""
    pmids = search_pubmed(query, max_results)
    return fetch_abstracts(pmids)

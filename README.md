# A.R.G.U.S. — AI Rapid Generation of Useful Strategy

A.R.G.U.S. is an early Market Access evidence workspace for Phase I, Phase II and Business Development decisions. It combines PubMed-based evidence discovery and LLM-assisted extraction with structured project inputs, strategic synthesis, evidence-gap tracking and exportable audit material.

## Dashboard

The Streamlit app now includes:

1. **Executive decision dashboard** — asset, indication, development stage, evidence cut-off, priority markets, access attractiveness, evidence maturity, Market Access thesis, value drivers, risks, minimum Target Access Profile (TAP), price/access scenarios and immediate actions.
2. **Disease, burden and unmet need** — disease definition, epidemiology, patient funnel, burden, unmet need and standard-of-care evidence categories.
3. **Competitive landscape** — marketed therapies, pipeline, differentiation and future comparator evidence categories.
4. **Regulatory and label** — approval precedent, label population, regulatory pathways and regulatory–HTA bridge categories.
5. **Trials and endpoints** — pivotal trials, endpoints, comparative feasibility and PICO/JCA readiness categories.
6. **HTA and reimbursement** — HTA decisions, access conditions, payer evidence and US access categories.
7. **Pricing and value corridor** — price benchmarks, economic value, budget impact and pricing dynamics categories.
8. **Real-world evidence** — data source inventory, natural history, treatment patterns, outcomes/burden, comparative options and RWE readiness categories.
9. **Strategic synthesis** — opportunity north star, payer value story, initial PICO, access strategy, evidence-generation plan and portfolio/BD implications.
10. **Evidence gaps and governance** — prioritized gap register, evidence status, confidence, manual source log and review cautions.
11. **Evidence annex** — JSON workspace export and CSV evidence-gap export.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your own credentials
streamlit run app.py
```

Required environment variables:
- `GROQ_API_KEY` — Groq API key used for structured extraction and synthesis.
- `ENTREZ_EMAIL` — email address required by NCBI Entrez.

Optional:
- `ENTREZ_API_KEY` — NCBI API key.
- `GROQ_MODEL` — model override; defaults to `openai/gpt-oss-120b`.
- `ARGUS_CACHE_FILE` — cache file path; defaults to `insight_cache.json`.

## Evidence and source limitations

- Literature retrieval currently uses **PubMed** for all seven modules. It is a discovery aid, not a complete search of regulatory, HTA, trial registry, reimbursement, pricing, claims/EHR or commercial pipeline databases.
- In particular, PubMed results alone cannot establish current FDA/EMA label status, HTA recommendations, reimbursement conditions, formulary coverage or public/net prices. Validate those against the relevant regulator, HTA body, trial registry, guideline, official price/formulary source or local expert.
- Price floor/target/ceiling, access scenarios, TAP, strategic thesis and gap register are structured analyst inputs; the application does not independently calculate or validate them.
- LLM output is a draft extraction/synthesis. Verify material claims against the linked source. Extraction confidence is not the same as certainty in the underlying evidence.
- The app distinguishes verified evidence, AI synthesis/interpretation, forecast assumptions and evidence gaps in its governance workflow. Preserve conflicting estimates and document cut-off, geography, population, comparator, outcome/timepoint and limitations.

## Current implementation boundary

The current release provides a shared PubMed + Groq retrieval workflow across the seven evidence modules, structured manual inputs for strategy and governance, session-based dashboard state, local JSON caching and downloadable JSON/CSV exports. Dedicated connectors to regulator, HTA, trial registry, pricing, RWE and commercial databases; formal epidemiology/patient-funnel calculations; economic modelling; ITC/NMA; and automated visualization are not implemented by this update.

## Cache and privacy

The local cache stores retrieved and generated evidence payloads in the configured JSON file. Treat that file as project data and protect it accordingly. The dashboard workspace itself is held in Streamlit session state; export it to retain the entered strategy and gap register beyond the session.

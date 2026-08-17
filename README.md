# ChemScholarAI — RAG for Computational Chemistry Research Articles

**ChemScholarAI is an end-to-end Retrieval-Augmented Generation (RAG) system designed to explore computational chemistry literature using natural-language questions.**

Instead of relying solely on an LLM's pretrained knowledge, ChemScholarAI retrieves relevant scientific papers from arXiv, grounds the response in those papers, and provides direct links to the underlying publications.

The central idea:

Don't ask an LLM remember the literature. Retrieve the literature first, then ask the LLM to reason over it.
ChemScholarAI addresses this by combining **scientific literature retrieval, domain-specific embeddings, metadata storage, vector search, and LLM-based generation** into one reproducible pipeline.

---

## What it can answer

* *Which computational methods are commonly used to study molecular potential energy surfaces?*
* *What applications of DFT have recently been reported for molecular systems?*
* *Which papers discuss ab initio molecular dynamics for a particular problem?*

The system does not attempt to replace scientific literature databases or researchers. Instead, it acts as a literature discovery and synthesis layer on top of a curated collection of computational chemistry papers.

---

## Architecture
ChemScholarAI separates the system into two main stages:

### 1. Scientific Literature Retrieval

### 2. Retrieval-Augmented Generation


```text
                         ┌─────────────────────┐
                         │      arXiv API      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Paper Discovery   │
                         │  + Metadata Fetch   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Computational       │
                         │ Chemistry Filter    │
                         └──────────┬──────────┘
                                    │
                           relevant papers
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
             ┌──────────────┐             ┌────────────────┐
             │ PostgreSQL   │             │    SPECTER2    │
             │              │             │                │
             │ Metadata     │             │ Paper vectors  │
             │ Authors      │             └───────┬────────┘
             │ Abstract     │                     |                     
             │ PDF URL      │                     ▼
             └──────────────┘             ┌────────────────┐
                                          │     FAISS      │
                                          │ Vector Search  │
                                          └───────┬────────┘
                                                  │
                                                  ▼
                                        Relevant papers
                                                  │
                         User question ──────────┘
                                                  │
                                                  ▼
                                        ┌────────────────┐
                                        │ RAG Pipeline   │
                                        │                │
                                        │  Retrieve      │
                                        │  Build context │
                                        │  Generate      │
                                        └───────┬────────┘
                                                │
                                                ▼
                                 Grounded LLM Answer + citations
                                                │
                                                ▼
        every interaction logged to Postgres (query, answer, sources, feedback)
```

---

Retrieval and metadata are deliberately split: **Postgres** owns the paper data and interaction logs, **FAISS** owns the embedding vectors, linked by a shared integer id(papers.id). This split also means the vector store can be swapped later without touching the metadata layer.

---

## Why SPECTER2?

A general-purpose sentence embedding model is not necessarily optimal for scientific literature.

ChemScholarAI uses **SPECTER2**, a scientific document representation model designed for scholarly documents.

Each paper is represented using information such as:

```text
Title
+
Abstract
```

The resulting embedding captures the paper's semantic representation and allows research questions to be compared against scientific publications in vector space.

---
## Project Structure

```text
.
├── app.py                        # Streamlit UI (Ask tab, feedback buttons)
├── store_embeddings.py           # Step 2 entry point: Postgres + FAISS ingestion
├── verify_pipeline.py            # diagnostics: row counts, round-trip test, sample query
├── config.py                     # typed settings, loaded from .env via pydantic-settings
├── filtering_topic.py            # keyword filter deciding which arXiv results are "comp chem"
├── fetch/
│   └── arxiv_fetch.py            # Step 1: arXiv API → filtered papers.jsonl
├── db/
│   ├── store.py                  # papers table: schema, upsert, connection helper
│   └── logs.py                   # interactions table: query/answer/feedback logging
├── embeddings/
│   ├── specter2_embedder.py      # SPECTER2 base model + paper/query adapters
│   └── faiss_store.py            # persistent FAISS index (cosine via normalized inner product)
├── rag/
│   └── rag.py                    # RAGPipeline: search → build_prompt → llm → rag
├── data/                         # generated at runtime (gitignored) — json/, faiss/
├── Dockerfile
├── docker-compose.yml            # postgres + app services
├── pyproject.toml / uv.lock      # dependencies (single source of truth, local and Docker)
├── .env.example                  # template — copy to .env and fill in
└── .env                          # secrets and config (gitignored, not committed)

```


## Technology Stack

| Component             | Technology               | Purpose                            |
| --------------------- | ------------------------ | ---------------------------------- |
| Literature source     | arXiv API                | Scientific paper discovery         |
| Filtering             | Python keyword matching  | Computational chemistry relevance  |
| Metadata database     | PostgreSQL               | Structured paper metadata + logs   |
| Scientific embeddings | SPECTER2                 | Scientific document representation |
| Vector database       | FAISS                    | Semantic similarity search         |
| LLM                   | GROQ API                 | Grounded answer generation         |
| UI                    | Streamlit                | Interactive research interface     |
| Configuration         | Pydantic Settings        | Typed environment configuration    |
| Containerization      | Docker                   | Reproducible deployment            |
| Package management    | uv                       | Python dependency management       |

---
## Current Data Pipeline

### 1. Literature Ingestion 

**Fetch** - queries the arXiv API (default categories `physics.chem-ph`, `cond-mat.mtrl-sci`), pulling title, abstract, authors, categories, dates, and PDF link. PDFs themselves aren't downloaded at this stage — the link is enough for a user to open the source.

---

### 2. Domain Filtering

**Filter** - a lightweight keyword filter (DFT, Hartree-Fock, ab initio, CCSD, molecular dynamics, potential energy surface, etc.) decides which results are actually computational chemistry, recording a `keyword_score` and `matched_keywords`. This stays separate from semantic retrieval so it can be swapped for a proper classifier later without touching anything else.

---

### 3. Metadata Storage

**Store** - filtered papers are upserted into PostgreSQL, keyed by `arxiv_id`, so repeated ingestion runs never create duplicates.

---

### 4. Scientific Embeddings

**Embed** — SPECTER2 converts each paper's `title + abstract` into a vector, stored in FAISS with the same integer id as its Postgres row. These vectors are stored in db for semantic similarity search.

---

### 5. Retrieval
**Retrieve** — a user's question is embedded with the same model (via a separate query-tuned adapter), compared against FAISS, and the top-k paper ids are hydrated back into full metadata from Postgres.

---

### 6. Generation
**Generate** — retrieved papers are assembled into a numbered context block (`[1]`, `[2]`, `[3]`...) and handed to the LLM with instructions to answer only from that context, citing sources inline.

---

## Prerequisites

- Python 3.11
- [uv] for dependency management
- Docker Desktop (for the containerized setup)
- A [Groq API key](https://console.groq.com) (free tier available)
- PostgreSQL

## Setup

### 1. Environment variables

Copy the template and fill in real values:

```bash
cp .env.example .env
```

| Variable | Purpose | Example |
|---|---|---|
| `ARXIV_SEARCH_CATEGORY` | Default arXiv query for Fetch | `cat:physics.chem-ph` |
| `ARXIV_MAX_RESULTS` | Papers per fetch run | `200` |
| `POSTGRES_HOST` / `PORT` / `DATABASE` / `USER` / `PASSWORD` | Postgres connection | see note below |
| `EMBEDDING_BATCH_SIZE` | SPECTER2 batching | `32` |
| `FAISS_INDEX_PATH` | Where the vector index is saved | `data/faiss/papers.index` |
| `GROQ_API_KEY` | Required for the Ask tab to generate answers | — |
| `GROQ_MODEL` | Groq model id | `openai/gpt-oss-120b` |
| `RAG_TOP_K` | Papers retrieved per question | `5` |

**Postgres user note:** the role name depends on how Postgres was installed. Homebrew's `postgresql@14` on macOS creates a superuser matching your OS username (not `postgres`) — check with `whoami`. The official Docker image, by contrast, creates whatever role name `POSTGRES_USER` specifies, so this is only a concern for local (non-Docker) development.

### 2a. Run locally with uv

```bash
uv sync
uv run python fetch/arxiv_fetch.py       # Step 1: fetch papers into papers.jsonl
uv run python store_embeddings.py        # Step 2: load into Postgres, embed, index in FAISS
uv run python verify_pipeline.py         # optional: sanity-check the pipeline
uv run streamlit run app.py
```

### 2b. Run with Docker

```bash
docker compose up --build -d
docker compose ps                        # confirm both services are Up/healthy

# one-off ingestion commands, run inside the app container:
docker compose exec app uv run python fetch/arxiv_fetch.py
docker compose exec app uv run python store_embeddings.py
```

Then open `http://localhost:8501`.

Postgres runs as its own service; `docker-compose.yml` overrides `POSTGRES_HOST` to `postgres` (the service name) for the app container, since `localhost` only applies outside the container network. The `./data` folder is bind-mounted, so `papers.jsonl` and the FAISS index persist across rebuilds.

## Usage

Type a question in the **Ask** tab (e.g. *"What are recent advances in ML interatomic potentials?"*) and submit. The answer streams back with inline `[1]`, `[2]`… citations; each cited paper is shown below in an expandable card with its abstract and a link to the PDF. Rate the answer with 👍/👎 — this is logged to Postgres (`interactions` table) along with the query, answer, sources, and response latency.


## Roadmap

- [ ] Hybrid search — sparse (BM25) + dense (SPECTER2) retrieval, fused via reciprocal rank fusion
- [ ] Query rewriting before retrieval
- [ ] FlashRank reranking between retrieval and generation
- [ ] Digest tab (recent papers / trending topics — TBD)
- [ ] Grafana dashboard on top of the `interactions` log table
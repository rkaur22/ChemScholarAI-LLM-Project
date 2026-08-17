# ChemScholarAI — RAG for Computational Chemistry Research Articles

ChemScholarAI is an end-to-end Retrieval Augmented Generation (RAG) system designed to help researchers explore computational chemistry literature using natural-language questions.
Instead of relying solely on an LLM's pretrained knowledge, ChemScholarAI retrieves relevant scientific papers from arXiv preprints, grounds the response in those papers, and provides direct links to the underlying publications.
Finding and synthesizing relevant computational chemistry literature is time-consuming, while general-purpose LLMs may produce answers that are difficult to verify or may rely on outdated knowledge. ChemScholarAI addresses this by combining scientific literature retrieval, domain-specific embeddings, metadata storage, vector search, and LLM-based generation into one reproducible pipeline.

## Project Aim
The goal of ChemScholarAI is to build a lightweight research assistant that can answer questions such as:
- Which computational methods are commonly used to study molecular potential energy surfaces?
- What applications of DFT have recently been reported for molecular systems?
- Which papers discuss ab initio molecular dynamics for a particular problem?
The system does not attempt to replace scientific literature databases or researchers. Instead, it acts as a literature discovery and synthesis layer on top of a curated collection of computational chemistry papers.

## How it works
ChemScholarAI separates the system into two main stages:
1. Scientific Literature Retrieval
Relevant computational chemistry papers are discovered from arXiv and filtered using domain-specific terminology.
The paper metadata is stored in PostgreSQL, while SPECTER2 generates scientific-document embeddings that are stored in a vector database.

2. Retrieval-Augmented Generation
When a user asks a question:
User question
      ↓
SPECTER2 query embedding
      ↓
Vector similarity search
      ↓
Relevant papers
      ↓
Metadata retrieved from PostgreSQL
      ↓
Context supplied to LLM
      ↓
Grounded answer with paper citations
The LLM therefore receives relevant scientific literature as context instead of having to answer entirely from its pretrained knowledge.


```
arXiv API ──▶ Fetch ──▶ Postgres (metadata)
                    └──▶ SPECTER2 embeddings ──▶ FAISS (vector index)

User question ──▶ RAGPipeline.rag()
                     ├─ search()        embed query (SPECTER2) → FAISS top-k → hydrate from Postgres
                     ├─ build_prompt()  numbered [1]/[2]/... context block
                     └─ llm()           Groq chat completion, cited answer

                  every call logged to Postgres (query, answer, sources, latency, 👍/👎)

Streamlit app ──▶ Ask tab (question in, cited answer + feedback buttons out)
```

Retrieval and metadata are deliberately split: **Postgres** owns paper metadata and interaction logs, **FAISS** owns the embedding vectors, linked by a shared integer id (`papers.id` doubles as the FAISS vector id — no separate mapping table needed).

## Project structure

```
.
├── app.py                        # Streamlit UI (Ask tab, feedback buttons)
├── store_embeddings.py           # Step 2 entry point: Postgres + FAISS ingestion
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
├── Dockerfile
├── docker-compose.yml            # postgres + app services
├── pyproject.toml / uv.lock      # dependencies (single source of truth, local and Docker)
└── .env                          # secrets and config (not committed — see below)
```

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker Desktop (for the containerized setup)
- A [Groq API key](https://console.groq.com) (free tier available)
- PostgreSQL, if running outside Docker (e.g. via Homebrew on macOS)

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

## Troubleshooting

- **`ModuleNotFoundError` for a local module** (`config`, `db`, etc.) when running a script directly — each entry-point script adds the project root to `sys.path` at the top of the file; make sure that line wasn't accidentally removed.
- **`role "postgres" does not exist`** — see the Postgres user note above; check `.env` matches your actual local role, and confirm with `env | grep POSTGRES` that no shell-exported variable is silently overriding `.env`.
- **`OMP: Error #179` on macOS** — a known fork/OpenMP conflict between torch and huggingface_hub's parallel downloader. `app.py` sets `OMP_NUM_THREADS=1` and related env vars before any torch import to work around it.
- **`streamlit: executable file not found`** in Docker — fixed by invoking `uv run streamlit ...` (or `python -m streamlit ...`) rather than the bare `streamlit` command, so it doesn't depend on `$PATH` resolving pip's console-script location.

## Roadmap

- [ ] Hybrid search — sparse (BM25) + dense (SPECTER2) retrieval, fused via reciprocal rank fusion
- [ ] Query rewriting before retrieval
- [ ] FlashRank reranking between retrieval and generation
- [ ] Digest tab (recent papers / trending topics — TBD)
- [ ] Grafana dashboard on top of the `interactions` log table
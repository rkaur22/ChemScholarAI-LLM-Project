"""
Step 2: Store — load Step 1's papers.jsonl into PostgreSQL, embed each
paper's title+abstract with SPECTER2, and persist the vectors in a FAISS
index keyed by the Postgres row id.

Call run_store_embeddings(...) directly, or from the Streamlit app — no CLI args.
"""

import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.append(str(Path(__file__).resolve().parent))
import config
from db import store
from embeddings.specter2_embedder import EMBEDDING_DIM, Specter2Embedder
from embeddings.faiss_store import FaissStore


def clean_paper(record: dict) -> dict:
    """Fill in fields that may be missing/None in the JSONL."""
    record = dict(record)
    if record.get("matched_keywords") is None:
        record["matched_keywords"] = []
    record.setdefault("keyword_score", 0)
    return record


def run_store_embeddings(
    jsonl_path: str,
    faiss_index_path: Optional[str] = None,
    force_reembed: bool = False,
    embed_batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """Load jsonl_path into Postgres, embed new/changed papers, update FAISS.

    Returns {"total": <rows upserted>, "embedded": <vectors newly written>}.
    progress_callback(str), if given, receives human-readable status lines
    instead of them going to stdout — handy for streaming into Streamlit.
    """

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    faiss_index_path = faiss_index_path or config.faiss_cfg.index_path
    embed_batch_size = embed_batch_size or config.embedding.batch_size

    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        raise FileNotFoundError(f"File not found: {jsonl_file}")

    conn = store.get_connection()
    store.create_tables(conn)

    log(f"Loading records from {jsonl_file} ...")
    to_embed = []  # (postgres_id, title, abstract)
    total = 0
    for record in store.load_jsonl(jsonl_file):
        paper = clean_paper(record)
        pg_id, already_embedded = store.upsert_paper(conn, paper)
        total += 1
        if force_reembed or not already_embedded:
            to_embed.append((pg_id, paper["title"], paper["abstract"]))

    log(f"Upserted {total} papers into Postgres. {len(to_embed)} need embedding.")

    if not to_embed:
        conn.close()
        log("Nothing new to embed.")
        return {"total": total, "embedded": 0}

    log("Loading SPECTER2 (base model + adapters) — first run downloads weights...")
    embedder = Specter2Embedder()
    faiss_store = FaissStore(dim=EMBEDDING_DIM, index_path=Path(faiss_index_path))

    ids = [row[0] for row in to_embed]
    titles = [row[1] for row in to_embed]
    abstracts = [row[2] for row in to_embed]

    log(f"Embedding {len(ids)} papers in batches of {embed_batch_size} ...")
    vectors = embedder.embed_papers(titles, abstracts, batch_size=embed_batch_size)

    faiss_store.add(ids, vectors)
    faiss_store.save()
    store.mark_embedded(conn, ids)
    conn.close()

    log(f"Saved {len(ids)} vectors to FAISS index at {faiss_index_path}")
    return {"total": total, "embedded": len(ids)}


if __name__ == "__main__":
    # Quick local smoke test without Streamlit — uses .env defaults.
    default_jsonl = Path(config.arxiv.output_dir) / config.arxiv.output_file
    print(run_store_embeddings(str(default_jsonl)))
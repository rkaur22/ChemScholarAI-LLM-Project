"""
db/store.py — PostgreSQL persistence for arXiv paper metadata (Step 2).

Creates the `papers` table if it doesn't exist, and provides upsert / fetch
helpers used by store_embeddings.py to load Step 1's JSONL output into
Postgres before embedding.
"""

import json
import sys
from pathlib import Path
from typing import Iterator

import psycopg2
import psycopg2.extras

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    id SERIAL PRIMARY KEY,
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT[] DEFAULT '{}',
    categories TEXT[] DEFAULT '{}',
    published_date TIMESTAMPTZ,
    updated_date TIMESTAMPTZ,
    pdf_url TEXT,
    keyword_score INTEGER DEFAULT 0,
    matched_keywords TEXT[] DEFAULT '{}',
    embedded BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

UPSERT_SQL = """
INSERT INTO papers (
    arxiv_id, title, abstract, authors, categories,
    published_date, updated_date, pdf_url,
    keyword_score, matched_keywords
) VALUES (
    %(arxiv_id)s, %(title)s, %(abstract)s, %(authors)s, %(categories)s,
    %(published_date)s, %(updated_date)s, %(pdf_url)s,
    %(keyword_score)s, %(matched_keywords)s
)
ON CONFLICT (arxiv_id) DO UPDATE SET
    title = EXCLUDED.title,
    abstract = EXCLUDED.abstract,
    authors = EXCLUDED.authors,
    categories = EXCLUDED.categories,
    published_date = EXCLUDED.published_date,
    updated_date = EXCLUDED.updated_date,
    pdf_url = EXCLUDED.pdf_url,
    keyword_score = EXCLUDED.keyword_score,
    matched_keywords = EXCLUDED.matched_keywords
RETURNING id, embedded;
"""


def get_connection():
    return psycopg2.connect(
        host=config.postgres.host,
        port=config.postgres.port,
        dbname=config.postgres.database,
        user=config.postgres.user,
        password=config.postgres.password,
    )


def create_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()


def load_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def upsert_paper(conn, paper: dict) -> tuple[int, bool]:
    """Insert or update one paper. Returns (postgres_id, already_embedded)."""
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(UPSERT_SQL, paper)
        row = cur.fetchone()
    conn.commit()
    return row["id"], row["embedded"]


def mark_embedded(conn, ids: list) -> None:
    if not ids:
        return
    with conn.cursor() as cur:
        cur.execute("UPDATE papers SET embedded = TRUE WHERE id = ANY(%s);", (ids,))
    conn.commit()


def fetch_unembedded(conn) -> list:
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("SELECT id, title, abstract FROM papers WHERE embedded = FALSE;")
        return [dict(r) for r in cur.fetchall()]
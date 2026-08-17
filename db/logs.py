"""
db/logs.py — Postgres logging for RAG interactions.

Every query, its answer, which papers it cited, how long it took, and
optional thumbs up/down feedback get a row here. This is the foundation
the rest of the roadmap builds on: hybrid search, query rewrite, and
reranking all eventually get *evaluated* against this log, and it's what
a future Grafana dashboard reads from directly.
"""

import sys
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS interactions (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    rewritten_query TEXT,
    answer TEXT NOT NULL,
    source_paper_ids INTEGER[] DEFAULT '{}',
    model TEXT,
    retrieval_k INTEGER,
    latency_ms INTEGER,
    feedback SMALLINT,  -- 1 = thumbs up, -1 = thumbs down, NULL = no feedback yet
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

INSERT_SQL = """
INSERT INTO interactions (
    query, rewritten_query, answer, source_paper_ids, model, retrieval_k, latency_ms
) VALUES (%s, %s, %s, %s, %s, %s, %s)
RETURNING id;
"""


def create_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()


def log_interaction(
    conn,
    query: str,
    answer: str,
    source_paper_ids: List[int],
    rewritten_query: Optional[str] = None,
    model: Optional[str] = None,
    retrieval_k: Optional[int] = None,
    latency_ms: Optional[int] = None,
) -> int:
    """Insert one interaction row. Returns its id, used to attach feedback later."""
    with conn.cursor() as cur:
        cur.execute(
            INSERT_SQL,
            (query, rewritten_query, answer, source_paper_ids, model, retrieval_k, latency_ms),
        )
        log_id = cur.fetchone()[0]
    conn.commit()
    return log_id


def set_feedback(conn, log_id: int, feedback: int) -> None:
    """feedback must be 1 (thumbs up) or -1 (thumbs down)."""
    if feedback not in (1, -1):
        raise ValueError("feedback must be 1 (up) or -1 (down)")
    with conn.cursor() as cur:
        cur.execute("UPDATE interactions SET feedback = %s WHERE id = %s;", (feedback, log_id))
    conn.commit()


def fetch_recent(conn, limit: int = 50) -> list:
    """Most recent interactions, newest first — handy for a future admin/debug view."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, query, answer, feedback, latency_ms, created_at FROM interactions "
            "ORDER BY created_at DESC LIMIT %s;",
            (limit,),
        )
        cols = ["id", "query", "answer", "feedback", "latency_ms", "created_at"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

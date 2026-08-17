import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from db import store, logs as db_logs
from embeddings.specter2_embedder import EMBEDDING_DIM, Specter2Embedder
from embeddings.faiss_store import FaissStore
from groq import Groq

SYSTEM_PROMPT = (
    """ You are a computational chemistry research scientist answering questions about computational
        chemistry and materials science particularly on electronic structure calculations and DFT. 
        Answer ONLY from the numbered paper excerpts provided. 
        If the excerpts don't contain enough information to answer, say so plainly 
        instead of guessing. Don't make up answers. When any question cannot be answered from the provided excerpts,
        respond with "I don't know" or "The provided excerpts do not contain enough information to answer this question."
    """
)

PROMPT_TEMPLATE = """Question: {query}

Relevant papers:
{context}

Answer the question using only the papers above, with inline [n] citations."""

class RAGPipeline:
    def __init__(self, top_k: Optional[int] = None):
        self.top_k = top_k or config.rag.top_k
        self.embedder = Specter2Embedder()
        self.faiss_store = FaissStore(dim=EMBEDDING_DIM, index_path=Path(config.faiss_cfg.index_path))
        self._groq_client: Optional[Groq] = None
        self._ensure_log_table()

    def _ensure_log_table(self) -> None:
        conn = store.get_connection()
        try:
            db_logs.create_tables(conn)
        finally:
            conn.close()

    def _get_groq_client(self) -> Groq:
        if self._groq_client is None:
            if not config.groq.api_key:
                raise RuntimeError("GROQ_API_KEY is not set — add it to your .env")
            self._groq_client = Groq(api_key=config.groq.api_key)
        return self._groq_client

    def search(self, query: str, k: Optional[int] = None) -> List[dict]:
        """Embed the query, search FAISS, hydrate hits from Postgres.

        Returns a list of dicts (id, arxiv_id, title, abstract, pdf_url,
        score), highest similarity first.
        """
        k = k or self.top_k
        query_vec = self.embedder.embed_query(query)
        ids, scores = self.faiss_store.search(query_vec, k=k)

        id_list = [i for i in ids if i != -1]
        if not id_list:
            return []

        conn = store.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, arxiv_id, title, abstract, pdf_url FROM papers WHERE id = ANY(%s);",
                    (id_list,),
                )
                cols = ["id", "arxiv_id", "title", "abstract", "pdf_url"]
                papers_by_id = {row[0]: dict(zip(cols, row)) for row in cur.fetchall()}
        finally:
            conn.close()

        results = []
        for pid, score in zip(ids, scores):
            if pid in papers_by_id:
                paper = dict(papers_by_id[pid])
                paper["score"] = round(float(score), 4)
                results.append(paper)
        return results

    def build_prompt(self, query: str, search_results: List[dict]) -> str:
        context = "\n\n".join(
            f"[{i}] Title: {p['title']}\nAbstract: {p['abstract']}"
            for i, p in enumerate(search_results, start=1)
        )
        return PROMPT_TEMPLATE.format(query=query, context=context)

    def llm(self, prompt: str) -> str:
        client = self._get_groq_client()
        completion = client.chat.completions.create(
            model=config.groq.model,
            temperature=config.groq.temperature,
            max_tokens=config.groq.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content

    def _log(self, query: str, answer: str, sources: List[dict], k: int, latency_ms: int) -> Optional[int]:
        """Best-effort logging — a logging failure should never break an
        otherwise-successful answer, so this swallows and returns None."""
        conn = store.get_connection()
        try:
            return db_logs.log_interaction(
                conn,
                query=query,
                answer=answer,
                source_paper_ids=[p["id"] for p in sources],
                model=config.groq.model,
                retrieval_k=k,
                latency_ms=latency_ms,
            )
        except Exception:
            return None
        finally:
            conn.close()

    def rag(self, query: str, k: Optional[int] = None) -> dict:
        if not query or not query.strip():
            raise ValueError("query must be non-empty")

        k = k or self.top_k
        start = time.monotonic()

        search_results = self.search(query, k=k)
        if not search_results:
            answer = "No relevant papers were found in the index for this query."
            sources: List[dict] = []
        else:
            prompt = self.build_prompt(query, search_results)
            answer = self.llm(prompt)
            sources = search_results

        latency_ms = int((time.monotonic() - start) * 1000)
        log_id = self._log(query, answer, sources, k, latency_ms)

        return {"answer": answer, "sources": sources, "log_id": log_id, "latency_ms": latency_ms}


# Module-level singleton so Streamlit doesn't reload SPECTER2 on every rerun.
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def answer_query(query: str, k: Optional[int] = None) -> dict:
    """Convenience wrapper around RAGPipeline.rag() for callers like app.py."""
    return get_pipeline().rag(query, k=k)


def submit_feedback(log_id: int, feedback: int) -> None:
    """feedback: 1 for thumbs up, -1 for thumbs down. Call with the log_id
    returned in answer_query()'s result dict."""
    conn = store.get_connection()
    try:
        db_logs.set_feedback(conn, log_id, feedback)
    finally:
        conn.close()
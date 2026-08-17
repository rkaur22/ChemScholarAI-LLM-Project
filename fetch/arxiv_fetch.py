"""
Fetch — arXiv abstract + metadata acquisition for computational chemistry RAG.

Scope (minimal):
- Source: arXiv API only (physics.chem-ph, cond-mat.mtrl-sci by default)
- Fetches: title, abstract, authors, categories, dates, pdf_url (link only)
- Output: JSONL file (one paper per line)
- Respects arXiv rate limits: max 1 request per 3 seconds, single connection
"""

import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
import filtering_topic

NAMESPACE = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list
    categories: list
    published_date: str
    updated_date: str
    pdf_url: str
    keyword_score: int = 0
    matched_keywords: list = None


def build_query_url(search_query: str, start: int, max_results: int) -> str:
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return f"{config.arxiv.api_base_url}?{urllib.parse.urlencode(params)}"


def fetch_page(search_query: str, start: int, max_results: int) -> str:
    url = build_query_url(search_query, start, max_results)
    req = urllib.request.Request(url, headers={"User-Agent": config.arxiv.user_agent})
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")


def parse_entries(xml_text: str) -> list:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", NAMESPACE):
        arxiv_id_full = entry.findtext("atom:id", default="", namespaces=NAMESPACE)
        arxiv_id = arxiv_id_full.rsplit("/", 1)[-1]  # strip URL prefix

        title = entry.findtext("atom:title", default="", namespaces=NAMESPACE).strip()
        abstract = entry.findtext("atom:summary", default="", namespaces=NAMESPACE).strip()

        authors = [
            a.findtext("atom:name", default="", namespaces=NAMESPACE)
            for a in entry.findall("atom:author", NAMESPACE)
        ]

        categories = [
            c.get("term") for c in entry.findall("atom:category", NAMESPACE) if c.get("term")
        ]

        published_date = entry.findtext("atom:published", default="", namespaces=NAMESPACE)
        updated_date = entry.findtext("atom:updated", default="", namespaces=NAMESPACE)

        pdf_url = ""
        for link in entry.findall("atom:link", NAMESPACE):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=" ".join(title.split()),
                abstract=" ".join(abstract.split()),
                authors=authors,
                categories=categories,
                published_date=published_date,
                updated_date=updated_date,
                pdf_url=pdf_url,
            )
        )
    return papers


def fetch_all(
    search_query: str,
    max_results: int,
    out_path: Path,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> int:
    """Fetch up to max_results papers, paginating in page_size chunks, respecting rate limits."""

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    total_written = 0
    total_seen = 0
    start = 0

    with out_path.open("w", encoding="utf-8") as f:
        while total_seen < max_results:
            page_size = min(config.arxiv.page_size, max_results - total_seen)
            log(f"Fetching results {start}..{start + page_size} for query: {search_query!r}")

            xml_text = fetch_page(search_query, start, page_size)
            papers = parse_entries(xml_text)

            if not papers:
                log("No more results returned — stopping.")
                break

            for paper in papers:
                relevant, score, matches = filtering_topic.is_computational_chemistry(
                    paper.title, paper.abstract, paper.categories
                )
                if relevant:
                    paper.keyword_score = score
                    paper.matched_keywords = matches
                    f.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")
                    total_written += 1

            total_seen += len(papers)
            start += len(papers)

            if total_seen < max_results:
                time.sleep(config.arxiv.rate_limit_delay)

    log(f"Matched and wrote {total_written} of {total_seen} papers seen.")
    return total_written


def fetch_and_save(
    query: Optional[str] = None,
    max_results: Optional[int] = None,
    out_path: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Fetch papers from arXiv matching `query` and write topic-filtered
    matches to a JSONL file. Falls back to .env / config defaults for any
    argument left as None. Returns the output file path as a string.
    """
    query = query or config.arxiv.search_category
    max_results = max_results or config.arxiv.max_results
    resolved_out = Path(out_path) if out_path else Path(config.arxiv.output_dir) / config.arxiv.output_file
    resolved_out.parent.mkdir(parents=True, exist_ok=True)

    count = fetch_all(query, max_results, resolved_out, progress_callback=progress_callback)

    msg = f"Done. Wrote {count} papers to {resolved_out.resolve()}"
    if progress_callback:
        progress_callback(msg)
    else:
        print(msg)

    return str(resolved_out)


if __name__ == "__main__":
    fetch_and_save()
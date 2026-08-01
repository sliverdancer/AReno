"""Run the frozen RIST-P0-v1.0 multi-database literature search."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

SEARCH_DATE = "2026-08-01"
USER_AGENT = "AReno-RIST-P0/1.0 (research audit; no contact data)"
QUERIES = (
    "multi-turn tool agent reward diversity credit assignment",
    "tool calling advantage collapse reward variation GRPO",
    "action span supervision all turn last turn tool agent",
    "function name argument supervision tool calling",
    "agent benchmark task difficulty reward resolution",
)


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8")


def _openalex(query: str) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "search": query,
            "filter": "from_publication_date:2023-01-01",
            "sort": "relevance_score:desc",
            "per-page": 50,
        }
    )
    return _request_json(f"https://api.openalex.org/works?{params}")


def _crossref(query: str) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "query.bibliographic": query,
            "filter": "from-pub-date:2023-01-01",
            "rows": 50,
            "select": "DOI,title,author,published,URL,type,container-title,abstract",
        }
    )
    return _request_json(f"https://api.crossref.org/works?{params}")


def _semantic_scholar(query: str) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "query": query,
            "limit": 50,
            "fields": "paperId,title,abstract,year,authors,url,venue,externalIds,citationCount",
        }
    )
    return _request_json(
        f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    )


def _arxiv(query: str) -> dict[str, Any]:
    terms = [term for term in query.split() if term]
    search_query = " AND ".join(f'all:"{term}"' for term in terms)
    params = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": 50,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    raw = _request_text(f"https://export.arxiv.org/api/query?{params}")
    root = ET.fromstring(raw)
    atom = "{http://www.w3.org/2005/Atom}"
    entries = []
    for entry in root.findall(f"{atom}entry"):
        entries.append(
            {
                "id": (entry.findtext(f"{atom}id") or "").strip(),
                "title": " ".join(
                    (entry.findtext(f"{atom}title") or "").split()
                ),
                "summary": " ".join(
                    (entry.findtext(f"{atom}summary") or "").split()
                ),
                "published": (entry.findtext(f"{atom}published") or "").strip(),
                "updated": (entry.findtext(f"{atom}updated") or "").strip(),
                "authors": [
                    (author.findtext(f"{atom}name") or "").strip()
                    for author in entry.findall(f"{atom}author")
                ],
            }
        )
    return {"query": query, "entries": entries, "raw_entry_count": len(entries)}


def run(output_dir: Path) -> dict[str, Any]:
    """Execute all frozen queries and retain successes and failures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "arxiv": _arxiv,
        "openalex": _openalex,
        "crossref": _crossref,
        "semantic_scholar": _semantic_scholar,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "RIST-P0-v1.0",
        "search_date": SEARCH_DATE,
        "queries": list(QUERIES),
        "sources": {},
    }
    for source_name, search in sources.items():
        source_records = []
        failures = []
        for query_index, query in enumerate(QUERIES):
            try:
                result = search(query)
                source_records.append({"query": query, "result": result})
            except Exception as exc:  # Evidence must preserve unavailable sources.
                failures.append(
                    {
                        "query": query,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            if query_index + 1 < len(QUERIES):
                time.sleep(1.0)
        path = output_dir / f"{source_name}.json"
        path.write_text(
            json.dumps(
                {"records": source_records, "failures": failures},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest["sources"][source_name] = {
            "path": str(path),
            "successful_queries": len(source_records),
            "failed_queries": len(failures),
        }
    manifest_path = output_dir / "search_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

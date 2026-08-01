"""Deduplicate and rank raw RIST-P0 literature-search records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

PRIMARY_TERMS = (
    "tool",
    "agent",
    "function call",
    "multi-turn",
    "multi turn",
)
SIGNAL_TERMS = (
    "reward",
    "credit",
    "advantage",
    "difficulty",
    "supervision",
    "process",
)
INTERACTION_TERMS = (
    "collapse",
    "diversity",
    "variation",
    "variance",
    "turn-level",
    "turn level",
    "action",
    "argument",
    "identif",
    "calibrat",
    "sample efficien",
)


def _text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_text(item) for item in value)
    return str(value or "")


def _openalex_abstract(inverted: Any) -> str:
    if not isinstance(inverted, dict):
        return ""
    positioned = []
    for token, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        positioned.extend((int(position), str(token)) for position in positions)
    return " ".join(token for _, token in sorted(positioned))


def _canonical_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _score(title: str, abstract: str) -> int:
    text = f"{title} {abstract}".lower()
    score = 0
    score += 3 * sum(term in text for term in PRIMARY_TERMS)
    score += 2 * sum(term in text for term in SIGNAL_TERMS)
    score += sum(term in text for term in INTERACTION_TERMS)
    if any(term in text for term in PRIMARY_TERMS) and any(
        term in text for term in SIGNAL_TERMS
    ):
        score += 5
    return score


def _records(raw_dir: Path) -> list[dict[str, Any]]:
    output = []
    for source in ("arxiv", "openalex", "crossref", "semantic_scholar"):
        payload = json.loads((raw_dir / f"{source}.json").read_text(encoding="utf-8"))
        for query_record in payload["records"]:
            query = query_record["query"]
            result = query_record["result"]
            if source == "arxiv":
                rows = result.get("entries", [])
                for row in rows:
                    output.append(
                        {
                            "source": source,
                            "query": query,
                            "id": row.get("id"),
                            "doi": None,
                            "title": row.get("title", ""),
                            "abstract": row.get("summary", ""),
                            "year": _text(row.get("published"))[:4],
                            "url": row.get("id"),
                        }
                    )
            elif source == "openalex":
                rows = result.get("results", [])
                for row in rows:
                    ids = row.get("ids") or {}
                    output.append(
                        {
                            "source": source,
                            "query": query,
                            "id": row.get("id"),
                            "doi": row.get("doi") or ids.get("doi"),
                            "title": row.get("title", ""),
                            "abstract": _openalex_abstract(
                                row.get("abstract_inverted_index")
                            ),
                            "year": row.get("publication_year"),
                            "url": (row.get("primary_location") or {}).get(
                                "landing_page_url"
                            ),
                        }
                    )
            elif source == "crossref":
                rows = (result.get("message") or {}).get("items", [])
                for row in rows:
                    title = _text(row.get("title"))
                    output.append(
                        {
                            "source": source,
                            "query": query,
                            "id": row.get("DOI"),
                            "doi": row.get("DOI"),
                            "title": title,
                            "abstract": _text(row.get("abstract")),
                            "year": _published_year(row.get("published")),
                            "url": row.get("URL"),
                        }
                    )
            else:
                rows = result.get("data", [])
                for row in rows:
                    external_ids = row.get("externalIds") or {}
                    output.append(
                        {
                            "source": source,
                            "query": query,
                            "id": row.get("paperId"),
                            "doi": external_ids.get("DOI"),
                            "title": row.get("title", ""),
                            "abstract": row.get("abstract", ""),
                            "year": row.get("year"),
                            "url": row.get("url"),
                        }
                    )
    return output


def _published_year(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    date_parts = value.get("date-parts")
    if not isinstance(date_parts, list) or not date_parts or not date_parts[0]:
        return None
    return int(date_parts[0][0])


def screen(raw_dir: Path, output_path: Path, *, limit: int) -> dict[str, Any]:
    """Write deterministic, title-deduplicated candidates for manual screening."""

    raw_records = _records(raw_dir)
    by_title: dict[str, dict[str, Any]] = {}
    for record in raw_records:
        canonical = _canonical_title(str(record["title"]))
        if not canonical:
            continue
        current = by_title.get(canonical)
        if current is None:
            current = {
                "title": record["title"],
                "abstract": record["abstract"],
                "year": record["year"],
                "doi": record["doi"],
                "url": record["url"],
                "sources": [],
                "queries": [],
            }
            by_title[canonical] = current
        if len(_text(record["abstract"])) > len(_text(current["abstract"])):
            current["abstract"] = record["abstract"]
        if not current.get("doi") and record.get("doi"):
            current["doi"] = record["doi"]
        if not current.get("url") and record.get("url"):
            current["url"] = record["url"]
        if record["source"] not in current["sources"]:
            current["sources"].append(record["source"])
        if record["query"] not in current["queries"]:
            current["queries"].append(record["query"])
    ranked = []
    for record in by_title.values():
        record["relevance_score"] = _score(
            _text(record["title"]), _text(record["abstract"])
        )
        ranked.append(record)
    ranked.sort(
        key=lambda row: (-int(row["relevance_score"]), _canonical_title(row["title"]))
    )
    raw_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(raw_dir.glob("*.json"))
    }
    result = {
        "schema_version": 1,
        "protocol": "RIST-P0-v1.0",
        "raw_record_count": len(raw_records),
        "deduplicated_title_count": len(ranked),
        "candidate_limit": limit,
        "raw_sha256": raw_hashes,
        "candidates": ranked[:limit],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args()
    result = screen(args.raw_dir, args.output, limit=args.limit)
    print(json.dumps({key: value for key, value in result.items() if key != "candidates"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the CPU-only BFCL v3 Base Multi-Turn feasibility audit package.

This script consumes a local mirror of allowlisted public BFCL files and writes
only derived metadata: file hashes, schema summaries, and a deterministic public
task-id selection. It intentionally does not copy raw BFCL prompts, answers, or
function documentation into the repository.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import re
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "external_audit" / "bfcl_v3_base_multiturn"

DEFAULT_SOURCE = pathlib.Path(os.environ.get("TEMP", ".")) / "bfcl_feasibility"
SOURCE = pathlib.Path(os.environ.get("BFCL_FEASIBILITY_SOURCE", DEFAULT_SOURCE))

TASK_FILE = "BFCL_v3_multi_turn_base.json"
POSSIBLE_ANSWER_FILE = "possible_answer_BFCL_v3_multi_turn_base.json"
FUNC_DOC_FILES = [
    "func_doc_gorilla_file_system.json",
    "func_doc_math_api.json",
    "func_doc_message_api.json",
    "func_doc_posting_api.json",
    "func_doc_ticket_api.json",
    "func_doc_trading_bot.json",
    "func_doc_travel_booking.json",
    "func_doc_vehicle_control.json",
]
OTHER_PUBLIC_FILES = ["README.md", "eval.yaml"]

SOURCE_URLS = {
    TASK_FILE: "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/data/BFCL_v3_multi_turn_base.json",
    POSSIBLE_ANSWER_FILE: "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/data/possible_answer/BFCL_v3_multi_turn_base.json",
    "README.md": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/README.md",
    "eval.yaml": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/eval.yaml",
    "func_doc_gorilla_file_system.json": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/data/multi_turn_func_doc/func_doc_gorilla_file_system.json",
    "func_doc_math_api.json": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/data/multi_turn_func_doc/func_doc_math_api.json",
    "func_doc_message_api.json": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/data/multi_turn_func_doc/func_doc_message_api.json",
    "func_doc_posting_api.json": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/data/multi_turn_func_doc/func_doc_posting_api.json",
    "func_doc_ticket_api.json": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/data/multi_turn_func_doc/func_doc_ticket_api.json",
    "func_doc_trading_bot.json": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/data/multi_turn_func_doc/func_doc_trading_bot.json",
    "func_doc_travel_booking.json": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/data/multi_turn_func_doc/func_doc_travel_booking.json",
    "func_doc_vehicle_control.json": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main/data/multi_turn_func_doc/func_doc_vehicle_control.json",
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_stream(path: pathlib.Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    out: list[Any] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        out.append(obj)
        idx = end
    return out


def natural_task_key(task_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.*?)(\d+)", task_id)
    if not match:
        return task_id, -1
    return match.group(1), int(match.group(2))


def ordered_counter(counter: collections.Counter[Any]) -> list[list[Any]]:
    return [[key, value] for key, value in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))]


def summarize_tasks(records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [r["id"] for r in records]
    selected = sorted(ids, key=natural_task_key)[:64]
    turn_counts = collections.Counter(len(r["question"]) for r in records)
    path_counts = collections.Counter(len(r["path"]) for r in records)
    class_counts: collections.Counter[str] = collections.Counter()
    tool_counts: collections.Counter[str] = collections.Counter()
    schema_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    bad_shape: list[str] = []
    for record in records:
        schema_counts[tuple(sorted(record.keys()))] += 1
        question = record.get("question")
        path = record.get("path")
        if not isinstance(question, list) or not isinstance(path, list):
            bad_shape.append(record.get("id", "<missing-id>"))
            continue
        for klass in record.get("involved_classes", []):
            class_counts[str(klass)] += 1
        for turn in path:
            for call in turn:
                tool_counts[str(call)] += 1
    return {
        "record_count": len(records),
        "unique_id_count": len(set(ids)),
        "first_id_natural": selected[0],
        "last_id_natural": sorted(ids, key=natural_task_key)[-1],
        "selected_task_count": len(selected),
        "selected_task_ids": selected,
        "turn_count_distribution": [[k, turn_counts[k]] for k in sorted(turn_counts)],
        "path_length_distribution": [[k, path_counts[k]] for k in sorted(path_counts)],
        "schema_keysets": [{"keys": list(k), "count": v} for k, v in schema_counts.items()],
        "involved_class_counts": ordered_counter(class_counts),
        "top_tool_counts": ordered_counter(tool_counts)[:20],
        "bad_shape_count": len(bad_shape),
        "bad_shape_ids": bad_shape[:20],
    }


def summarize_answers(records: list[dict[str, Any]]) -> dict[str, Any]:
    schema_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    answer_turn_counts: collections.Counter[int] = collections.Counter()
    answer_call_counts: collections.Counter[int] = collections.Counter()
    ids = []
    for record in records:
        ids.append(record.get("id"))
        schema_counts[tuple(sorted(record.keys()))] += 1
        ground_truth = record.get("ground_truth", [])
        if isinstance(ground_truth, list):
            answer_turn_counts[len(ground_truth)] += 1
            for turn in ground_truth:
                if isinstance(turn, list):
                    answer_call_counts[len(turn)] += 1
    return {
        "record_count": len(records),
        "unique_id_count": len(set(ids)),
        "schema_keysets": [{"keys": list(k), "count": v} for k, v in schema_counts.items()],
        "answer_turn_count_distribution": [[k, answer_turn_counts[k]] for k in sorted(answer_turn_counts)],
        "answer_calls_per_turn_distribution": [[k, answer_call_counts[k]] for k in sorted(answer_call_counts)],
    }


def summarize_func_doc(path: pathlib.Path) -> dict[str, Any]:
    records = load_json_stream(path)
    if len(records) == 1 and isinstance(records[0], list):
        records = records[0]
    schema_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    names = []
    for record in records:
        if isinstance(record, dict):
            schema_counts[tuple(sorted(record.keys()))] += 1
            names.append(record.get("name") or record.get("function", {}).get("name"))
    return {
        "file": path.name,
        "record_count": len(records),
        "schema_keysets": [{"keys": list(k), "count": v} for k, v in schema_counts.items()],
        "function_names": [str(name) for name in names if name][:10],
    }


def main() -> None:
    required = [TASK_FILE, POSSIBLE_ANSWER_FILE, *FUNC_DOC_FILES, *OTHER_PUBLIC_FILES]
    missing = [name for name in required if not (SOURCE / name).exists()]
    if missing:
        raise SystemExit(f"Missing BFCL feasibility source files in {SOURCE}: {missing}")

    OUT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "audit_name": "BFCL-v3-base-multiturn-public-feasibility",
        "stage": "CPU-only feasibility; no model inference; no training; no held-out access",
        "source_root_used_for_build": str(SOURCE),
        "files": [
            {
                "name": name,
                "bytes": (SOURCE / name).stat().st_size,
                "sha256": sha256_file(SOURCE / name),
                "source_url": SOURCE_URLS.get(name),
            }
            for name in required
        ],
    }

    task_records = load_json_stream(SOURCE / TASK_FILE)
    answer_records = load_json_stream(SOURCE / POSSIBLE_ANSWER_FILE)
    task_summary = summarize_tasks(task_records)
    answer_summary = summarize_answers(answer_records)
    func_doc_summary = [summarize_func_doc(SOURCE / name) for name in FUNC_DOC_FILES]
    task_ids = task_summary["selected_task_ids"]

    schema = {
        "task_file": TASK_FILE,
        "possible_answer_file": POSSIBLE_ANSWER_FILE,
        "json_format": "whitespace-separated JSON object stream",
        "task_summary": {k: v for k, v in task_summary.items() if k != "selected_task_ids"},
        "possible_answer_summary": answer_summary,
        "function_doc_summary": func_doc_summary,
        "deterministic_selection_rule": "natural numeric sort by task id; first 64 public task ids; no model outcomes used",
    }

    (OUT / "PUBLIC_FILE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "SCHEMA_SUMMARY.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "SELECTED_TASK_IDS.txt").write_text("\n".join(task_ids) + "\n", encoding="utf-8")

    report = f"""# BFCL v3 Base Multi-Turn CPU-Only Feasibility Audit

Status: `PASS_FOR_PROTOCOL_DESIGN_ONLY`

Scope:

- Public BFCL v3 Base Multi-Turn files only.
- No model inference, no training, no GPU, no API calls.
- No sealed or held-out split access.
- Raw BFCL prompts, possible answers, and function docs are not copied into this repository.

Key source facts:

- Task records: {task_summary['record_count']}
- Unique task ids: {task_summary['unique_id_count']}
- Possible-answer records: {answer_summary['record_count']}
- Function-doc files: {len(func_doc_summary)}
- JSON format: whitespace-separated JSON object stream.
- Deterministic candidate subset: first {len(task_ids)} ids after natural numeric task-id sort.

Feasibility conclusion:

BFCL v3 Base Multi-Turn is feasible as an external public audit target for the
reward-resolution collapse paper. The public split provides multi-turn tasks,
oracle answers, and function documentation sufficient to define a pre-inference
audit protocol. The next step must be a separate frozen execution protocol that
binds source file hashes, the selected ids, model revisions, decoding settings,
and terminal finalization rules before any model request is sent.

Important limitation:

This audit does not establish reward-resolution evidence. It only establishes
that a public, non-held-out BFCL split can be used for an outcome-blind external
audit. Any reward-resolution result requires a newly frozen execution receipt and
separate authorization for inference.

Selection rule:

`natural_numeric_sort(task_id)[:64]`

This rule is outcome-blind: it uses task ids only and does not inspect model
responses or reward outcomes.

Artifacts:

- `PUBLIC_FILE_MANIFEST.json`: public source URLs, byte sizes, SHA-256 hashes.
- `SCHEMA_SUMMARY.json`: derived schema/statistical summary only.
- `SELECTED_TASK_IDS.txt`: deterministic public task-id subset.
"""
    (OUT / "FEASIBILITY_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    readme = """# BFCL v3 Base Multi-Turn External Audit Feasibility

This directory contains CPU-only, outcome-blind metadata for using the public
BFCL v3 Base Multi-Turn split as an external audit target.

Do not place raw BFCL prompt, answer, or function-documentation files here.
Rebuild with:

```bash
python research/reward_identifiability_supervision_topology/negative_result/build_bfcl_feasibility_audit.py
```

Set `BFCL_FEASIBILITY_SOURCE` if the public files are mirrored outside the
default temp directory.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()

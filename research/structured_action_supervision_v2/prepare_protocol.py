"""Freeze the SAS v2 tool-readiness datasets, cells, and selection rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "research"
    / "structured_action_supervision_v1"
    / "stages"
    / "Q0"
    / "dataset"
    / "train.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "research"
    / "structured_action_supervision_v2"
    / "stages"
    / "B1"
)
PROTOCOL_ID = "SAS-TR-v2.0"
SPLIT_SALT = "SAS-TR-v2.0-tool-readiness-split"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _row_key(row: dict[str, Any]) -> str:
    material = (
        f"{SPLIT_SALT}\x1f{row['id']}\x1f{row['constraint_signature']}"
    ).encode("utf-8")
    return _sha256_bytes(material)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic 16/16/16 calibration, validation, and reserve splits."""

    if len(rows) != 48:
        raise ValueError(f"expected exactly 48 source training rows, found {len(rows)}")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("source row ids must be unique")
    if len({row.get("constraint_signature") for row in rows}) != len(rows):
        raise ValueError("source constraint signatures must be unique")
    ordered = sorted((dict(row) for row in rows), key=_row_key)
    return {
        "calibration": ordered[:16],
        "validation": ordered[16:32],
        "reserve": ordered[32:],
    }


def select_eligible_cell(cell_results: dict[str, dict[str, float]]) -> str | None:
    """Select the smallest non-thinking budget that passes every frozen gate."""

    for cell_id in ("N128", "N512"):
        result = cell_results.get(cell_id)
        if result is None:
            continue
        if (
            result.get("first_turn_executable_rate", 0.0) >= 0.95
            and result.get("four_turn_completion_rate", 0.0) >= 0.75
            and 0.05 <= result.get("positive_reward_rate", 0.0) <= 0.95
            and result.get("fabricated_call_count", 1.0) == 0
        ):
            return cell_id
    return None


def prepare(source: Path, output_dir: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    splits = split_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for name, split in splits.items():
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in split
        ).encode("utf-8")
        destination = output_dir / f"{name}.jsonl"
        destination.write_bytes(payload)
        files[name] = {
            "path": destination.name,
            "rows": len(split),
            "sha256": _sha256_bytes(payload),
            "ids": [row["id"] for row in split],
            "constraint_signatures": [
                row["constraint_signature"] for row in split
            ],
        }

    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage": "B1-CPU-FREEZE",
        "source": {
            "path": _display_path(source),
            "sha256": _sha256_bytes(source.read_bytes()),
            "rows": len(rows),
        },
        "split_salt": SPLIT_SALT,
        "files": files,
        "heldout_consumed": False,
        "model": {
            "checkpoint": "Qwen/Qwen3-0.6B",
            "weights_sha256": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        },
        "cells": {
            "D128": {
                "thinking_mode": "tokenizer_default",
                "max_new_tokens": 128,
                "role": "diagnostic_only",
            },
            "D512": {
                "thinking_mode": "tokenizer_default",
                "max_new_tokens": 512,
                "role": "diagnostic_only",
            },
            "N128": {
                "thinking_mode": "disabled",
                "max_new_tokens": 128,
                "role": "eligible_interface",
            },
            "N512": {
                "thinking_mode": "disabled",
                "max_new_tokens": 512,
                "role": "eligible_interface",
            },
        },
        "sampling_seeds": [3101, 3202],
        "trajectories_per_cell": 32,
        "selection_order": ["N128", "N512"],
        "calibration_gates": {
            "first_turn_executable_rate_min": 0.95,
            "four_turn_completion_rate_min": 0.75,
            "positive_reward_rate_min": 0.05,
            "positive_reward_rate_max": 0.95,
            "fabricated_call_count_max": 0,
        },
        "decision_rule": (
            "Select N128 if it passes every calibration gate; otherwise select "
            "N512 if it passes; otherwise KILL_QWEN3_0_6B_INSTRUMENT. D128 and "
            "D512 diagnose the failure mechanism and are never eligible."
        ),
        "validation_rule": (
            "Run only the selected interface on validation.jsonl with both frozen "
            "seeds. It must independently pass the same gates. Reserve remains unopened."
        ),
        "claim_boundary": (
            "B1/B2 qualify an inference interface. They do not estimate AF versus "
            "LF efficacy and cannot reopen the main-conference route."
        ),
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "manifest.json").write_text(manifest_payload, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare(args.source.resolve(), args.output_dir.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

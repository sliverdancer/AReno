"""Forensic audit of the consumed SAS Q1 v1.1 rollout evidence."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_ROOT = (
    REPO_ROOT
    / "research"
    / "structured_action_supervision_v1"
    / "stages"
    / "Q1_v1_1"
    / "attempt_20260730"
)


def _setting(config: dict[str, Any], key: str) -> Any:
    for section in config.get("settings", {}).get("sections", []):
        for item in section.get("items", []):
            if item.get("key") == key:
                return item.get("value")
    raise KeyError(f"missing resolved config setting: {key}")


def audit_evidence(evidence_root: Path) -> dict[str, Any]:
    """Summarize whether Q1 v1.1 responses exhausted thinking budgets."""

    pattern = str(evidence_root / "runs" / "*" / "metrics" / "rollout_samples.*.jsonl")
    sample_paths = [Path(path) for path in sorted(glob.glob(pattern))]
    rows: list[dict[str, Any]] = []
    per_run: dict[str, dict[str, Any]] = {}

    for sample_path in sample_paths:
        config_paths = sorted(sample_path.parent.glob("areno_run_config.*.json"))
        if len(config_paths) != 1:
            raise ValueError(
                f"expected one resolved config beside {sample_path}, found {len(config_paths)}"
            )
        config = json.loads(config_paths[0].read_text(encoding="utf-8"))
        max_new_tokens = int(_setting(config, "max_new_tokens"))
        thinking = _setting(config, "chat_template_enable_thinking")
        run_rows = [
            json.loads(line)
            for line in sample_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        run_key = sample_path.parents[1].name
        for row in run_rows:
            row["_run_key"] = run_key
            row["_max_new_tokens"] = max_new_tokens
            row["_thinking"] = thinking
        rows.extend(run_rows)
        per_run[run_key] = {
            "raw_samples": len(run_rows),
            "max_new_tokens": max_new_tokens,
            "chat_template_enable_thinking": thinking,
            "starts_with_think": sum(_starts_with_think(row) for row in run_rows),
            "closed_think": sum(_closed_think(row) for row in run_rows),
            "budget_saturated": sum(_budget_saturated(row) for row in run_rows),
            "parsed_tool_calls": sum(bool(row.get("tool_calls")) for row in run_rows),
        }

    total = len(rows)
    counts = {
        "rollout_files": len(sample_paths),
        "raw_samples": total,
        "starts_with_think": sum(_starts_with_think(row) for row in rows),
        "closed_think": sum(_closed_think(row) for row in rows),
        "budget_saturated": sum(_budget_saturated(row) for row in rows),
        "contains_tool_markup": sum(_contains_tool_markup(row) for row in rows),
        "parsed_tool_calls": sum(bool(row.get("tool_calls")) for row in rows),
        "single_assistant_turn": sum(_assistant_turn_count(row) == 1 for row in rows),
        "unique_responses": len({str(row.get("final_answer", "")) for row in rows}),
    }
    loss_mask_true_distribution = Counter(
        int(row.get("loss_mask_true", -1)) for row in rows
    )
    complete = len(sample_paths) == 4 and total == 128
    root_cause_candidate = bool(
        complete
        and counts["starts_with_think"] == total
        and counts["closed_think"] == 0
        and counts["budget_saturated"] == total
        and counts["contains_tool_markup"] == 0
        and counts["parsed_tool_calls"] == 0
        and counts["single_assistant_turn"] == total
    )
    return {
        "schema_version": 1,
        "source_protocol": "SAS-P0-v1.1",
        "stage": "B0-RETROSPECTIVE-FORENSICS",
        "decision": (
            "PASS_ROOT_CAUSE_CANDIDATE_THINK_BUDGET_EXHAUSTION"
            if root_cause_candidate
            else "BLOCK_ROOT_CAUSE_NOT_ISOLATED"
        ),
        "counts": counts,
        "loss_mask_true_distribution": {
            str(key): value for key, value in sorted(loss_mask_true_distribution.items())
        },
        "per_run": per_run,
        "gate_checks": {
            "complete_consumed_evidence": complete,
            "all_responses_enter_thinking": counts["starts_with_think"] == total,
            "no_response_closes_thinking": counts["closed_think"] == 0,
            "all_responses_saturate_generation_budget": counts["budget_saturated"]
            == total,
            "no_raw_tool_markup": counts["contains_tool_markup"] == 0,
            "no_parsed_tool_calls": counts["parsed_tool_calls"] == 0,
            "exactly_one_assistant_turn_per_trajectory": counts[
                "single_assistant_turn"
            ]
            == total,
        },
        "interpretation": (
            "The evidence is consistent with generation-budget exhaustion inside "
            "Qwen3 thinking mode before a tool call could be emitted."
        ),
        "causal_boundary": (
            "The stored samples do not include an API finish_reason, so this audit "
            "does not by itself prove that thinking mode caused the failure. A fresh "
            "prespecified thinking-mode by token-budget factorial is required."
        ),
    }


def _answer(row: dict[str, Any]) -> str:
    return str(row.get("final_answer") or "")


def _starts_with_think(row: dict[str, Any]) -> bool:
    return _answer(row).lstrip().startswith("<think>")


def _closed_think(row: dict[str, Any]) -> bool:
    return "</think>" in _answer(row)


def _budget_saturated(row: dict[str, Any]) -> bool:
    return int(row.get("loss_mask_true", -1)) == int(row["_max_new_tokens"])


def _contains_tool_markup(row: dict[str, Any]) -> bool:
    answer = _answer(row)
    markers = (
        "<tool_call>",
        "</tool_call>",
        '"name": "search_catalog"',
        '"name":"search_catalog"',
    )
    return any(marker in answer for marker in markers)


def _assistant_turn_count(row: dict[str, Any]) -> int:
    return sum(
        isinstance(message, dict) and message.get("role") == "assistant"
        for message in row.get("messages", [])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_evidence(args.evidence_root.resolve())
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())

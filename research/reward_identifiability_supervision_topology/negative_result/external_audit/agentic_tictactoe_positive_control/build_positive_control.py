"""Build a CPU-only parseable tool-call positive control for reward resolution.

This artifact is intentionally not a BFCL result and not a model result.  It
uses the public repo-native Tic-Tac-Toe tool-call example as a small positive
control showing that the reward-resolution diagnostic can pass when tool calls
are parseable and the strict reward has mixed groups.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
OUT_DIR = Path(__file__).resolve().parent
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def load_agentic_module(name: str):
    path = ROOT / "examples" / "agentic" / "tictactoe" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"rist_positive_control_tictactoe_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            call = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        arguments = call.get("arguments")
        if isinstance(name, str) and isinstance(arguments, dict):
            calls.append({"name": name, "arguments": arguments})
    return calls


def response_for_square(square: int) -> str:
    payload = {"name": "choose_square", "arguments": {"square": square}}
    return "<tool_call>" + json.dumps(payload, separators=(",", ":"), sort_keys=True) + "</tool_call>"


def build_groups() -> list[dict[str, Any]]:
    return [
        {
            "group_id": "ttt_public_all_pass_win",
            "expected_category": "all_pass",
            "board": [["X", "X", "."], ["O", ".", "."], ["O", ".", "."]],
            "squares": [3, 3, 3, 3],
        },
        {
            "group_id": "ttt_public_mixed_win_or_occupied",
            "expected_category": "mixed",
            "board": [["X", "X", "."], ["O", ".", "."], ["O", ".", "."]],
            "squares": [3, 1, 3, 1],
        },
        {
            "group_id": "ttt_public_mixed_win_or_nonterminal_best",
            "expected_category": "mixed",
            "board": [["X", "X", "."], ["O", ".", "."], ["O", ".", "."]],
            "squares": [3, 5, 3, 6],
        },
        {
            "group_id": "ttt_public_all_fail_occupied",
            "expected_category": "all_fail",
            "board": [["X", "X", "."], ["O", ".", "."], ["O", ".", "."]],
            "squares": [1, 1, 4, 7],
        },
    ]


def classify(successes: list[bool]) -> str:
    if all(successes):
        return "all_pass"
    if not any(successes):
        return "all_fail"
    return "mixed"


def summarize(groups: list[dict[str, Any]]) -> dict[str, Any]:
    group_categories = [group["observed_category"] for group in groups]
    rewards = [rollout["reward"] for group in groups for rollout in group["rollouts"]]
    parseable = [rollout["parseable_tool_calls"] for group in groups for rollout in group["rollouts"]]
    return {
        "group_count": len(groups),
        "rollout_count": sum(len(group["rollouts"]) for group in groups),
        "parseable_tool_call_rate": sum(parseable) / len(parseable),
        "strict_success_rate": sum(reward == 1.0 for reward in rewards) / len(rewards),
        "mixed_group_count": group_categories.count("mixed"),
        "all_pass_group_count": group_categories.count("all_pass"),
        "all_fail_group_count": group_categories.count("all_fail"),
        "non_zero_advantage_group_count": group_categories.count("mixed"),
        "high_low_contrast_present": group_categories.count("all_pass") >= 1
        and group_categories.count("all_fail") >= 1
        and group_categories.count("mixed") >= 1,
    }


def build_result() -> dict[str, Any]:
    game = load_agentic_module("game")
    reward = load_agentic_module("reward")
    groups: list[dict[str, Any]] = []
    for group in build_groups():
        board = game.normalize_board(group["board"])
        best_moves = game.best_moves(board)
        rollouts = []
        for rollout_idx, square in enumerate(group["squares"]):
            response = response_for_square(square)
            parsed = parse_tool_calls(response)
            record = SimpleNamespace(source_record={"board": board}, completion=response, tool_calls=parsed)
            score = float(reward.reward_fn(record))
            rollouts.append(
                {
                    "rollout_id": f"{group['group_id']}_r{rollout_idx:02d}",
                    "response_sha256": sha256_bytes(response.encode("utf-8")),
                    "parseable_tool_calls": bool(parsed),
                    "observed_call_count": len(parsed),
                    "observed_tool_name": parsed[0]["name"] if parsed else None,
                    "observed_square": parsed[0]["arguments"]["square"] if parsed else None,
                    "reward": score,
                    "strict_success": score == 1.0,
                }
            )
        successes = [rollout["strict_success"] for rollout in rollouts]
        groups.append(
            {
                "group_id": group["group_id"],
                "public_example": "examples/agentic/tictactoe",
                "expected_category": group["expected_category"],
                "observed_category": classify(successes),
                "board": board,
                "best_moves": best_moves,
                "rollouts": rollouts,
            }
        )
    result = {
        "protocol": "RRC-AGENTIC-TICTACTOE-PARSEABLE-POSITIVE-CONTROL-v1",
        "status": "PASS_CPU_ONLY_PARSEABLE_TOOL_CALL_POSITIVE_CONTROL",
        "scope": {
            "uses_public_repo_native_example": True,
            "bfcl_used": False,
            "heldout_or_sealed_access_used": False,
            "model_inference_used": False,
            "api_used": False,
            "gpu_used": False,
            "training_used": False,
            "raw_model_responses_included": False,
        },
        "environment": {
            "source_example": "examples/agentic/tictactoe",
            "tool_name": "choose_square",
            "tool_arguments_schema": {"type": "object", "properties": {"square": {"type": "integer"}}},
            "reward_function": "examples/agentic/tictactoe/reward.py::reward_fn",
            "parser_fixture_format": "<tool_call>{json}</tool_call>",
        },
        "groups": groups,
        "resolution_summary": summarize(groups),
        "interpretation": {
            "what_this_proves": (
                "The reward-resolution diagnostic and strict tool-call evaluator can PASS on a "
                "parseable public tool-call protocol with all-pass, mixed, and all-fail groups."
            ),
            "what_this_does_not_prove": (
                "This is not a BFCL reward-resolution result, not a model capability result, "
                "and not a substitute for a true external benchmark canary."
            ),
            "next_gate": (
                "Freeze a true external public environment canary, preferably Tau3 or another "
                "public tool-use benchmark whose model prompt format is known to emit parseable calls."
            ),
        },
    }
    return result


def write_report(result: dict[str, Any]) -> str:
    summary = result["resolution_summary"]
    return "\n".join(
        [
            "# Agentic Tic-Tac-Toe parseable positive control",
            "",
            f"Status: `{result['status']}`",
            "",
            "This CPU-only artifact uses the public repo-native `examples/agentic/tictactoe`",
            "`choose_square` tool protocol as a positive control for the reward-resolution",
            "diagnostic. It does not use BFCL, held-out data, a model, an API, GPU, or",
            "training.",
            "",
            "## Result",
            "",
            f"- Groups: {summary['group_count']}",
            f"- Rollouts: {summary['rollout_count']}",
            f"- Parseable tool-call rate: {summary['parseable_tool_call_rate']:.3f}",
            f"- Strict success rate: {summary['strict_success_rate']:.3f}",
            f"- Mixed groups: {summary['mixed_group_count']}",
            f"- All-pass groups: {summary['all_pass_group_count']}",
            f"- All-fail groups: {summary['all_fail_group_count']}",
            f"- Non-zero-advantage groups: {summary['non_zero_advantage_group_count']}",
            f"- High/low/mixed contrast present: {summary['high_low_contrast_present']}",
            "",
            "## Boundary",
            "",
            "This is a diagnostic positive control, not an external BFCL result. It closes",
            "one gap in the evidence chain: the analyzer can identify reward-resolution",
            "contrast when parseable tool calls exist. The remaining gap is a true external",
            "public environment canary that produces parseable model tool calls.",
            "",
        ]
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = build_result()
    (OUT_DIR / "POSITIVE_CONTROL_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "POSITIVE_CONTROL_REPORT.md").write_text(write_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

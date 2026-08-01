"""Generate deterministic CPU evidence for the ARCA P1 production audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from areno.api.agentic import (
    LossMaskPolicy,
    ResponseSpan,
    RolloutSession,
    _chat_response_message_tool_calls,
)
from areno.api.rewards import RewardEvent, RewardRecord, load_reward_fn

from research.silent_reward_contracts.auditor import (
    canonical_action_arguments,
    masks_are_distinguishable,
)


SCHEMA_VERSION = "arca.p1.v1"
CASE_FIELDS = (
    "case_id",
    "failure_family",
    "severity",
    "ordinary_exit_ok",
    "observed_value",
    "oracle_value",
    "detected",
    "contract_status",
    "source_path",
)


def _source_record() -> dict[str, Any]:
    return {
        "task_id": "arca-p1-fixed",
        "weights": [2, -2, 1, -1],
        "threshold": 0,
    }


def _production_events() -> list[RewardEvent]:
    actions = (1, 0, 1, 0)
    events = []
    for index, bit in enumerate(actions):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": f"call-{index}",
                                "type": "function",
                                "function": {
                                    "name": "choose_bit",
                                    "arguments": {"bit": bit},
                                },
                            }
                        ]
                    }
                }
            ]
        }
        normalized = _chat_response_message_tool_calls(response)
        function = normalized[0]["function"]
        events.append(
            RewardEvent(
                type="assistant_tool_call",
                name=function["name"],
                arguments=function["arguments"],
            )
        )
    return events


def _record(events: list[RewardEvent]) -> RewardRecord:
    return RewardRecord(
        prompt="Choose four bits.",
        completion="done",
        source_record=_source_record(),
        trace=events,
    )


def _canonical_events(events: list[RewardEvent]) -> list[RewardEvent]:
    return [
        event.model_copy(
            update={"arguments": canonical_action_arguments(event.arguments)}
        )
        for event in events
    ]


def _mask(mode: str) -> list[bool]:
    session = RolloutSession(
        None,
        sampling_params=None,
        loss_mask_policy=LossMaskPolicy(trainable_turns=mode),
    )
    sample = SimpleNamespace(
        response_spans=[
            ResponseSpan("assistant_tool_call", 2),
            ResponseSpan("assistant_text", 2),
        ],
        response_tokens=[10, 11, 20, 21],
        response_kind="assistant_text",
        loss_mask_override=[True, True, True, True],
        token_row=[],
        response_mask_row=[],
        loss_mask_row=[],
    )
    session._apply_trainable_turn_mode(sample)
    return list(sample.loss_mask_override)


def build_evidence(repo_root: Path) -> dict[str, Any]:
    reward_path = repo_root / "examples/agentic/care_bifurcation/reward.py"
    reward_fn = load_reward_fn(str(reward_path))
    production_events = _production_events()
    observed_reward = float(reward_fn(_record(production_events)))
    oracle_reward = float(reward_fn(_record(_canonical_events(production_events))))

    masks = {
        mode: _mask(mode)
        for mode in ("all_assistant", "last_assistant", "final_answer")
    }
    alias = not masks_are_distinguishable(
        masks["last_assistant"], masks["final_answer"]
    )
    argument_types = sorted(
        {type(event.arguments).__name__ for event in production_events}
    )
    cases = [
        {
            "case_id": "ARCA-P1-F1-STRINGIFIED-ARGUMENTS",
            "failure_family": "F1_serialization_type",
            "severity": "high",
            "ordinary_exit_ok": True,
            "observed_value": observed_reward,
            "oracle_value": oracle_reward,
            "detected": observed_reward != oracle_reward,
            "contract_status": "FAIL_SILENT_REWARD_CORRUPTION",
            "source_path": "areno/api/agentic.py::"
            "_chat_response_message_tool_calls -> "
            "examples/agentic/care_bifurcation/reward.py::reward_fn",
        },
        {
            "case_id": "ARCA-P1-F5-ISSUE199-ALIAS",
            "failure_family": "F5_treatment_identity",
            "severity": "informational",
            "ordinary_exit_ok": True,
            "observed_value": int(alias),
            "oracle_value": 1,
            "detected": alias,
            "contract_status": "PASS_DECLARED_EQUIVALENCE_CONTROL",
            "source_path": "examples/agentic/trainable_turns_ablation/README.md",
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "ARCA-CPU-AUDIT-v0.1",
        "gpu_executed": False,
        "production_argument_types": argument_types,
        "masks": masks,
        "cases": cases,
    }


def write_evidence(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    evidence = build_evidence(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "production_contract_cases.json"
    csv_path = output_dir / "production_contract_cases.csv"
    json_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(evidence["cases"])
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    write_evidence(args.output_dir, args.repo_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

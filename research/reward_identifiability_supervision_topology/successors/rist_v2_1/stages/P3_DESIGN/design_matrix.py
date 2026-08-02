"""Generate the CPU-only RIST-v2.1 factorial training design."""

from __future__ import annotations

from itertools import product
from typing import Any

FAMILIES = {
    "qwen3": {
        "checkpoint": "Qwen/Qwen3-0.6B",
        "capacity_gate_required": True,
    },
    "gemma4": {
        "checkpoint": "google/gemma-4-E2B-it",
        "capacity_gate_required": True,
    },
}
ALGORITHMS = ("gspo", "grpo")
ARMS = {
    "AF": {
        "trainable_turns": "all_assistant",
        "tool_call_supervision": "full",
        "content_claim": "full_call",
    },
    "LF": {
        "trainable_turns": "last_assistant",
        "tool_call_supervision": "full",
        "content_claim": "full_call",
    },
    "AN": {
        "trainable_turns": "all_assistant",
        "tool_call_supervision": "name_only",
        "content_claim": "name_only",
    },
    "LN": {
        "trainable_turns": "last_assistant",
        "tool_call_supervision": "name_only",
        "content_claim": "name_only",
    },
}
PAIRED_SEEDS = (7101, 7202, 7303)


def build_matrix() -> list[dict[str, Any]]:
    """Return the frozen 2x2x4x3 step-matched run matrix."""

    rows = []
    for family, algorithm, arm, seed in product(
        FAMILIES, ALGORITHMS, ARMS, PAIRED_SEEDS
    ):
        treatment = ARMS[arm]
        rows.append(
            {
                "run_id": f"{family}-{algorithm}-{arm}-{seed}",
                "family": family,
                "checkpoint": FAMILIES[family]["checkpoint"],
                "algorithm": algorithm,
                "arm": arm,
                "seed": seed,
                "trainable_turns": treatment["trainable_turns"],
                "tool_call_supervision": treatment["tool_call_supervision"],
                "content_claim": treatment["content_claim"],
                "dataset_role": "train",
                "group_size": 8,
                "step_matched": True,
                "token_common_support_required": True,
                "capacity_gate_required": True,
                "execution_authorized": False,
            }
        )
    return rows


def validate_matrix(rows: list[dict[str, Any]]) -> None:
    """Reject missing cells, unpaired seeds, or accidental execution authority."""

    expected = {
        (family, algorithm, arm, seed)
        for family, algorithm, arm, seed in product(
            FAMILIES, ALGORITHMS, ARMS, PAIRED_SEEDS
        )
    }
    observed = {
        (row["family"], row["algorithm"], row["arm"], row["seed"])
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("factorial matrix must contain exactly 48 unique runs")
    for row in rows:
        if row["dataset_role"] != "train":
            raise ValueError("P3 may use only the future train split")
        if row["execution_authorized"] is not False:
            raise ValueError("CPU design must not authorize execution")
        expected_treatment = ARMS[str(row["arm"])]
        if row["trainable_turns"] != expected_treatment["trainable_turns"]:
            raise ValueError("trainable-turn treatment mismatch")
        if row["tool_call_supervision"] != expected_treatment["tool_call_supervision"]:
            raise ValueError("tool-call supervision treatment mismatch")
        if row["content_claim"] != expected_treatment["content_claim"]:
            raise ValueError("content-treatment claim mismatch")


def cli_treatment_args(row: dict[str, Any]) -> list[str]:
    """Return only the public treatment/algorithm/seed CLI arguments."""

    arguments = [
        "--algo",
        str(row["algorithm"]),
        "--seed",
        str(row["seed"]),
        "--trainable-turns",
        str(row["trainable_turns"]),
    ]
    arguments.extend(
        ["--tool-call-supervision", str(row["tool_call_supervision"])]
    )
    return arguments

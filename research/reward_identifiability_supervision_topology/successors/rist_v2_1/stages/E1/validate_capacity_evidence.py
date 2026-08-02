"""Validate future per-checkpoint RIST serving and one-step training evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REQUIRED_ALGORITHMS = {"gspo", "grpo"}


def validate_capacity(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return fail-closed capacity gates for one checkpoint/GPU pairing."""

    serving = evidence["serving_canary"]
    training = evidence["training_canaries"]
    if {str(row["algorithm"]) for row in training} != REQUIRED_ALGORITHMS:
        raise ValueError("capacity evidence requires exactly GSPO and GRPO canaries")
    total_memory = float(evidence["gpu_total_memory_gib"])
    peak_memory = max(
        [float(serving["peak_memory_gib"])]
        + [float(row["peak_memory_gib"]) for row in training]
    )
    if not math.isfinite(total_memory) or total_memory <= 0.0:
        raise ValueError("gpu_total_memory_gib must be positive and finite")
    if not math.isfinite(peak_memory) or peak_memory < 0.0:
        raise ValueError("peak memory must be non-negative and finite")

    serving_pass = all(
        (
            serving.get("health_pass") is True,
            int(serving.get("task_count", 0)) == 32,
            int(serving.get("complete_four_turn_count", 0)) == 32,
            int(serving.get("raw_response_count", 0)) == 128,
            serving.get("oom") is False,
            serving.get("retry_count") == 0,
        )
    )
    algorithm_pass = {
        str(row["algorithm"]): all(
            (
                row.get("optimizer_step_completed") is True,
                int(row.get("trainable_tokens", 0)) > 0,
                math.isfinite(float(row.get("loss", float("nan")))),
                math.isfinite(float(row.get("gradient_norm", float("nan")))),
                row.get("oom") is False,
                row.get("checkpoint_roundtrip") is True,
            )
        )
        for row in training
    }
    memory_headroom_pass = peak_memory / total_memory <= 0.85
    identity_pass = all(
        isinstance(evidence.get(field), str) and bool(evidence[field])
        for field in ("checkpoint", "model_revision", "tokenizer_sha256", "gpu_name")
    )
    passed = (
        identity_pass
        and serving_pass
        and all(algorithm_pass.values())
        and memory_headroom_pass
    )
    return {
        "checkpoint": evidence.get("checkpoint"),
        "gpu_name": evidence.get("gpu_name"),
        "identity_pass": identity_pass,
        "serving_pass": serving_pass,
        "algorithm_pass": algorithm_pass,
        "peak_memory_gib": peak_memory,
        "memory_headroom_fraction": 1.0 - peak_memory / total_memory,
        "memory_headroom_pass": memory_headroom_pass,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    result = validate_capacity(evidence)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

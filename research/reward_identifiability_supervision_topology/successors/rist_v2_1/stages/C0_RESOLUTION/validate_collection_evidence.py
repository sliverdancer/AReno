"""Fail-closed validation for one completed C0 family/split collection job."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate_collection(
    manifest: dict[str, Any],
    family: str,
    split: str,
    result: dict[str, Any],
    journal_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    journal_sha256: str,
) -> dict[str, Any]:
    """Verify completeness and identity without interpreting reward outcomes."""

    if manifest.get("protocol") != "RIST-C0-v2.1":
        raise ValueError("unexpected C0 manifest")
    if family not in manifest["models"] or split not in manifest["splits"]:
        raise ValueError("unknown C0 family or split")
    spec = manifest["splits"][split]
    expected_pairs = {
        (str(task["id"]), int(seed))
        for task in source_rows
        for seed in spec["rollout_seeds"]
    }
    signatures = {str(task["id"]): str(task["task_signature"]) for task in source_rows}
    trajectories = result.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError("C0 result is missing trajectories")
    observed_pairs = [
        (str(row.get("task_id")), int(row.get("rollout_seed")))
        for row in trajectories
    ]
    duplicate_pairs = len(observed_pairs) != len(set(observed_pairs))
    identity_complete = set(observed_pairs) == expected_pairs and not duplicate_pairs
    reward_schema_pass = all(
        type(row.get("strict_success")) is int
        and row["strict_success"] in (0, 1)
        and row.get("split") == split
        and str(row.get("task_signature")) == signatures.get(str(row.get("task_id")))
        and 1 <= int(row.get("raw_response_count", 0)) <= 4
        for row in trajectories
    )

    journal_by_pair: dict[tuple[str, int], list[int]] = defaultdict(list)
    journal_schema_pass = True
    for row in journal_rows:
        try:
            pair = (str(row["task_id"]), int(row["rollout_seed"]))
            turn_index = int(row["turn_index"])
        except (KeyError, TypeError, ValueError):
            journal_schema_pass = False
            continue
        if (
            pair not in expected_pairs
            or str(row.get("task_signature")) != signatures.get(pair[0])
            or not isinstance(row.get("raw_response"), dict)
        ):
            journal_schema_pass = False
        journal_by_pair[pair].append(turn_index)
    contiguous_journal_pass = journal_schema_pass and all(
        sorted(journal_by_pair.get(pair, []))
        == list(range(int(trajectory["raw_response_count"])))
        for pair, trajectory in zip(observed_pairs, trajectories)
    )
    expected_journal_count = sum(
        int(row.get("raw_response_count", 0)) for row in trajectories
    )
    journal_count_pass = len(journal_rows) == expected_journal_count
    result_contract_pass = all(
        (
            result.get("protocol") == "RIST-C0-v2.1",
            result.get("family") == family,
            result.get("checkpoint") == manifest["models"][family],
            result.get("split") == split,
            result.get("split_sha256") == spec["sha256"],
            result.get("complete") is True,
            result.get("infrastructure_error") is None,
            result.get("retry_count") == 0,
            int(result.get("expected_trajectory_count", -1)) == len(expected_pairs),
            int(result.get("trajectory_count", -1)) == len(expected_pairs),
            int(result.get("collection_concurrency", -1))
            == int(manifest["collection_concurrency"][family]),
            result.get("raw_journal_sha256") == journal_sha256,
            (
                manifest.get("runtime_identity_required") is not True
                or (
                    isinstance(result.get("runtime_identity"), dict)
                    and result["runtime_identity"].get("family") == family
                    and result["runtime_identity"].get("checkpoint")
                    == manifest["models"][family]
                    and result["runtime_identity"].get("model_revision")
                    == manifest["model_revisions"][family]
                    and result["runtime_identity"].get("tokenizer_snapshot_sha256")
                    == manifest["tokenizer_snapshot_sha256"][family]
                    and result["runtime_identity"].get("source_commit")
                    == manifest["source_commit"]
                    and _is_sha256(result["runtime_identity"].get("model_weights_sha256"))
                    and _is_sha256(
                        result["runtime_identity"].get("snapshot_verification_sha256")
                    )
                    and float(
                        result["runtime_identity"].get("gpu_total_memory_gib", 0.0)
                    )
                    > 0.0
                    and all(
                        isinstance(result["runtime_identity"].get(field), str)
                        and bool(result["runtime_identity"][field])
                        for field in (
                            "gpu_name",
                            "gpu_uuid",
                            "driver_version",
                            "cuda_version",
                            "torch_version",
                            "endpoint",
                        )
                    )
                )
            ),
        )
    )
    passed = all(
        (
            result_contract_pass,
            identity_complete,
            reward_schema_pass,
            contiguous_journal_pass,
            journal_count_pass,
        )
    )
    return {
        "protocol": "RIST-C0-EVIDENCE-VALIDATION-v1",
        "family": family,
        "split": split,
        "expected_trajectory_count": len(expected_pairs),
        "trajectory_count": len(trajectories),
        "journal_response_count": len(journal_rows),
        "result_contract_pass": result_contract_pass,
        "identity_complete": identity_complete,
        "duplicate_task_seed_identity": duplicate_pairs,
        "reward_schema_pass": reward_schema_pass,
        "contiguous_journal_pass": contiguous_journal_pass,
        "journal_count_pass": journal_count_pass,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family", choices=("qwen3", "gemma4"), required=True)
    parser.add_argument("--split", choices=("calibration", "qualification"), required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    split_spec = manifest["splits"][args.split]
    source_path = Path(manifest["source_data_dir"]) / split_spec["file"]
    if _sha256(source_path) != split_spec["sha256"]:
        raise ValueError("C0 source split hash mismatch")
    result = validate_collection(
        manifest,
        args.family,
        args.split,
        json.loads(args.result.read_text()),
        _read_jsonl(args.journal),
        _read_jsonl(source_path),
        _sha256(args.journal),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

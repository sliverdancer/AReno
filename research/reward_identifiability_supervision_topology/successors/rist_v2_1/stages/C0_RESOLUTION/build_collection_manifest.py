"""Build non-executable pretraining resolution-collection jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
SOURCE_DATA = (
    REPO_ROOT
    / "research/reward_identifiability_supervision_topology/successors/rist_v2"
    / "stages/D2/data"
)
COLLECTOR = Path(__file__).with_name("collect_pretraining.py")
MODELS = {
    "qwen3": "Qwen/Qwen3-0.6B",
    "gemma4": "google/gemma-4-E2B-it",
}
ROLLOUT_SEEDS = list(range(11001, 11033))


def build_manifest(output_root: Path) -> dict[str, Any]:
    source = json.loads((SOURCE_DATA / "manifest.json").read_text())
    splits = {
        split: {
            "file": source["files"][split]["file"],
            "sha256": source["files"][split]["sha256"],
            "task_count": source["files"][split]["count"],
            "rollout_seeds": ROLLOUT_SEEDS,
            "trajectory_count": source["files"][split]["count"] * len(ROLLOUT_SEEDS),
        }
        for split in ("calibration", "qualification")
    }
    jobs = []
    for family, checkpoint in MODELS.items():
        for split in ("calibration", "qualification"):
            job_id = f"{family}-{split}"
            output = output_root / "c0" / job_id / "result.json"
            journal = output_root / "c0" / job_id / "raw_responses.jsonl"
            ledger = (
                output_root / "c0" / "qualification_ledgers" / f"{family}.json"
                if split == "qualification"
                else None
            )
            command = [
                "python3",
                str(COLLECTOR),
                "--manifest",
                "{C0_MANIFEST}",
                "--family",
                family,
                "--split",
                split,
                "--base-url",
                "http://127.0.0.1:{PORT}/v1",
                "--output",
                str(output),
                "--journal",
                str(journal),
            ]
            if ledger is not None:
                command.extend(["--ledger", str(ledger)])
            jobs.append(
                {
                    "job_id": job_id,
                    "family": family,
                    "checkpoint": checkpoint,
                    "split": split,
                    "trajectory_count": splits[split]["trajectory_count"],
                    "client_command_template": command,
                    "ledger": None if ledger is None else str(ledger),
                    "execution_authorized": False,
                }
            )
    return {
        "protocol": "RIST-C0-v2.1",
        "source_protocol": source["protocol"],
        "source_data_dir": str(SOURCE_DATA),
        "group_size": 8,
        "groups_per_task": 4,
        "splits": splits,
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 128,
            "retry_count": 0,
        },
        "models": MODELS,
        "job_count": len(jobs),
        "execution_authorized": False,
        "commands_are_templates_only": True,
        "parent_heldout_opened": False,
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.output_root)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

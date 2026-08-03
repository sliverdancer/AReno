"""Build non-executable pretraining resolution-collection jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
SOURCE_DATA = (
    REPO_ROOT
    / "research/reward_identifiability_supervision_topology/successors/rist_v2"
    / "stages/D2/data"
)
COLLECTOR = (
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/"
    "stages/C0_RESOLUTION/collect_pretraining.py"
)
SOURCE_DATA_RELATIVE = (
    "research/reward_identifiability_supervision_topology/successors/rist_v2/"
    "stages/D2/data"
)
MODELS = {
    "qwen3": "Qwen/Qwen3-0.6B",
    "gemma4": "google/gemma-4-E2B-it",
}
MODEL_REVISIONS = {
    "qwen3": "c1899de289a04d12100db370d81485cdf75e47ca",
    "gemma4": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
}
TOKENIZER_SNAPSHOT_SHA256 = {
    "qwen3": "c83c7f983e1204841852a4cb47cff31dfd829437c80dccc55dd52d0c8fe532b1",
    "gemma4": "7c813a44e67aa09d81001db777c261d858417b45ce0204bc6a32f0b7b96720f7",
}
# Concurrency changes only scheduling. Request seeds are derived from the frozen
# rollout seed, task signature, and turn index, and retries remain forbidden.
COLLECTION_CONCURRENCY = {"qwen3": 8, "gemma4": 4}
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
                COLLECTOR,
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
                "--runtime-identity",
                f"{{{family.upper()}_RUNTIME_IDENTITY_JSON}}",
            ]
            if ledger is not None:
                command.extend(["--ledger", str(ledger)])
            jobs.append(
                {
                    "job_id": job_id,
                    "family": family,
                    "checkpoint": checkpoint,
                    "model_revision": MODEL_REVISIONS[family],
                    "tokenizer_snapshot_sha256": TOKENIZER_SNAPSHOT_SHA256[family],
                    "collection_concurrency": COLLECTION_CONCURRENCY[family],
                    "split": split,
                    "trajectory_count": splits[split]["trajectory_count"],
                    "client_command_template": command,
                    "ledger": None if ledger is None else str(ledger),
                    "execution_authorized": False,
                }
            )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "protocol": "RIST-C0-v2.1",
        "source_commit": source_commit,
        "source_protocol": source["protocol"],
        "source_data_dir": SOURCE_DATA_RELATIVE,
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
        "model_revisions": MODEL_REVISIONS,
        "tokenizer_snapshot_sha256": TOKENIZER_SNAPSHOT_SHA256,
        "collection_concurrency": COLLECTION_CONCURRENCY,
        "runtime_identity_required": True,
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

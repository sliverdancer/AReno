"""Build non-executable D4 checkpoint-evaluation job templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
EVALUATOR = (
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/"
    "stages/D4_EVAL/evaluate_checkpoint.py"
)
DATA_DIR = (
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/"
    "stages/D4_EVAL/data"
)


def _job(
    *,
    job_id: str,
    checkpoint_path: str,
    split: str,
    output_root: Path,
    prerequisite: str,
) -> dict[str, Any]:
    output = output_root / "evaluations" / job_id / "result.json"
    journal = output_root / "evaluations" / job_id / "raw_responses.jsonl"
    ledger = (
        output_root / "confirmatory_ledgers" / f"{job_id}.json"
        if split == "confirmatory"
        else None
    )
    client = [
        "python3",
        EVALUATOR,
        "--base-url",
        "http://127.0.0.1:{PORT}/v1",
        "--data-dir",
        DATA_DIR,
        "--split",
        split,
        "--checkpoint-id",
        job_id,
        "--output",
        str(output),
        "--journal",
        str(journal),
    ]
    if ledger is not None:
        client.extend(["--ledger", str(ledger)])
    return {
        "job_id": job_id,
        "checkpoint_path": checkpoint_path,
        "split": split,
        "prerequisite": prerequisite,
        "serve_command_template": [
            "areno",
            "serve",
            "--model-path",
            checkpoint_path,
            "--tp-size",
            "1",
            "--world-size",
            "1",
            "--port",
            "{PORT}",
        ],
        "client_command_template": client,
        "output": str(output),
        "journal": str(journal),
        "ledger": None if ledger is None else str(ledger),
        "execution_authorized": False,
    }


def build_evaluation_manifest(
    training_manifest: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    steps = [int(step) for step in training_manifest["saved_checkpoint_steps"]]
    if steps != [25, 50, 75, 100]:
        raise ValueError("D4 requires frozen checkpoints at 25, 50, 75, and 100")
    jobs = []
    families: dict[str, str] = {}
    for run in training_manifest["runs"]:
        families[str(run["family"])] = str(run["checkpoint"])
        run_id = str(run["run_id"])
        checkpoint_root = output_root / "runs" / run_id / "checkpoints"
        for step in steps:
            jobs.append(
                _job(
                    job_id=f"{run_id}-dev-step-{step:03d}",
                    checkpoint_path=str(checkpoint_root / f"step_{step:06d}"),
                    split="dev_curve",
                    output_root=output_root,
                    prerequisite="TRAIN_RUN_COMPLETE_AND_CHECKPOINT_HASHED",
                )
            )
        jobs.append(
            _job(
                job_id=f"{run_id}-confirmatory-step-100",
                checkpoint_path=str(checkpoint_root / "step_000100"),
                split="confirmatory",
                output_root=output_root,
                prerequisite="ALL_TRAINING_AND_ANALYSIS_HASHES_FROZEN",
            )
        )
    for family, checkpoint in sorted(families.items()):
        jobs.append(
            _job(
                job_id=f"{family}-base-dev-step-000",
                checkpoint_path=checkpoint,
                split="dev_curve",
                output_root=output_root,
                prerequisite="E1_MODEL_CAPACITY_AND_TOKENIZER_PASS",
            )
        )
    if len(jobs) != 242 or len({job["job_id"] for job in jobs}) != 242:
        raise ValueError("D4 requires 192 dev, 48 confirmatory, and 2 base jobs")
    return {
        "protocol": "RIST-D4-EVAL-v2.1",
        "training_manifest_source_commit": training_manifest["source_commit"],
        "job_count": len(jobs),
        "development_job_count": sum(job["split"] == "dev_curve" for job in jobs),
        "confirmatory_job_count": sum(job["split"] == "confirmatory" for job in jobs),
        "execution_authorized": False,
        "commands_are_templates_only": True,
        "confirmatory_generated_after_ledger": True,
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training_manifest = json.loads(args.training_manifest.read_text())
    result = build_evaluation_manifest(training_manifest, args.output_root)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

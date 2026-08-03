"""Build fail-closed one-step GSPO/GRPO capacity job templates."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

MODELS = {
    "qwen3": {
        "checkpoint": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "tokenizer_snapshot_sha256": "c83c7f983e1204841852a4cb47cff31dfd829437c80dccc55dd52d0c8fe532b1",
        "model_path": "{QWEN3_MODEL_PATH}",
        "gpu_pairing": "{QWEN3_GPU_UUID}",
        "minimum_gpu_memory_gib": 24,
    },
    "gemma4": {
        "checkpoint": "google/gemma-4-E2B-it",
        "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "tokenizer_snapshot_sha256": "7c813a44e67aa09d81001db777c261d858417b45ce0204bc6a32f0b7b96720f7",
        "model_path": "{GEMMA4_MODEL_PATH}",
        "gpu_pairing": "{GEMMA4_LARGER_MEMORY_GPU_UUID}",
        "minimum_gpu_memory_gib": 48,
    },
}
ALGORITHMS = ("gspo", "grpo")
TRAINING_SEED = 8101
REPO_ROOT = Path(__file__).resolve().parents[6]
MONITOR = (
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/"
    "stages/E1/monitor_gpu_command.py"
)
SERVING_CLIENT = (
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/"
    "stages/E1/run_serving_canary.py"
)
METRICS_EXTRACTOR = (
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/"
    "stages/E1/extract_one_step_metrics.py"
)
CHECKPOINT_MANIFEST = (
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/"
    "stages/E1/build_checkpoint_manifest.py"
)
DIRECTORY_MANIFEST = (
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/"
    "stages/E1/build_directory_manifest.py"
)


def _command(family: str, algorithm: str, run_root: Path) -> list[str]:
    model = MODELS[family]
    run_id = f"{family}-{algorithm}-AF-{TRAINING_SEED}"
    root = run_root / run_id
    return [
        "areno",
        "train",
        "--ckpt",
        str(model["model_path"]),
        "--dataset-path",
        "{E1_HIGH_RESOLUTION_CANARY_JSONL}",
        "--dataset-loader-fn",
        "examples/agentic/rist_v2_1/dataset_loader.py",
        "--reward-fn-path",
        "examples/agentic/rist_v2_1/reward.py",
        "--agent-fn",
        "examples/agentic/rist_v2_1/run_agent.py",
        "--algo",
        algorithm,
        "--seed",
        str(TRAINING_SEED),
        "--tp-size",
        "1",
        "--world-size",
        "1",
        "--batch-size",
        "1",
        "--n-samples",
        "8",
        "--mini-bs",
        "8",
        "--max-running-prompts",
        "8",
        "--max-prompt-tokens",
        "1024",
        "--max-new-tokens",
        "128",
        "--max-context-len",
        "4096",
        "--agent-timeout-s",
        "900",
        "--attn-backend",
        "native",
        "--drop-rollout-state",
        "--save-path",
        str(root / "checkpoints"),
        "--save-interval",
        "1",
        "--max-steps",
        "1",
        "--metrics-log-dir",
        str(root / "metrics"),
        "--trainable-turns",
        "all_assistant",
        "--tool-call-supervision",
        "full",
    ]


def build_manifest(run_root: Path) -> dict[str, Any]:
    serving_canaries = []
    jobs = []
    for family, model in MODELS.items():
        serving_root = run_root / "serving" / family
        serve_command = [
            "areno",
            "serve",
            "--model-path",
            str(model["model_path"]),
            "--tp-size",
            "1",
            "--world-size",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--max-running-prompts",
            "1",
            "--default-max-tokens",
            "128",
            "--eager-decode",
            "--attn-backend",
            "native",
            "--disable-thinking",
        ]
        serving_canaries.append(
            {
                "family": family,
                "checkpoint": model["checkpoint"],
                "gpu_pairing": model["gpu_pairing"],
                "minimum_gpu_memory_gib": model["minimum_gpu_memory_gib"],
                "monitored_serve_command_template": [
                    "python3",
                    MONITOR,
                    "--gpu-uuid",
                    str(model["gpu_pairing"]),
                    "--output",
                    str(serving_root / "gpu_monitor.json"),
                    "--log",
                    str(serving_root / "serve.log"),
                    "--",
                    *serve_command,
                ],
                "client_command_template": [
                    "python3",
                    SERVING_CLIENT,
                    "--manifest",
                    "{E1_MANIFEST}",
                    "--family",
                    family,
                    "--runtime-identity",
                    f"{{{family.upper()}_RUNTIME_IDENTITY_JSON}}",
                    "--task-limit",
                    "32",
                    "--base-url",
                    "http://127.0.0.1:8000/v1",
                    "--journal",
                    str(serving_root / "raw_responses.jsonl"),
                    "--output",
                    str(serving_root / "result.json"),
                ],
                "serve_log": str(serving_root / "serve.log"),
                "execution_authorized": False,
            }
        )
        for algorithm in ALGORITHMS:
            run_id = f"{family}-{algorithm}-AF-{TRAINING_SEED}"
            root = run_root / run_id
            train_command = _command(family, algorithm, run_root)
            checkpoint = root / "checkpoints" / "step_000001"
            jobs.append(
                {
                    "run_id": run_id,
                    "family": family,
                    "checkpoint": model["checkpoint"],
                    "model_revision": model["revision"],
                    "tokenizer_snapshot_sha256": model["tokenizer_snapshot_sha256"],
                    "gpu_pairing": model["gpu_pairing"],
                    "minimum_gpu_memory_gib": model["minimum_gpu_memory_gib"],
                    "algorithm": algorithm,
                    "arm": "AF",
                    "trainable_turns": "all_assistant",
                    "tool_call_supervision": "full",
                    "seed": TRAINING_SEED,
                    "max_steps": 1,
                    "save_interval": 1,
                    "command_template": train_command,
                    "monitored_command_template": [
                        "python3",
                        MONITOR,
                        "--gpu-uuid",
                        str(model["gpu_pairing"]),
                        "--output",
                        str(root / "gpu_monitor.json"),
                        "--log",
                        str(root / "train.log"),
                        "--",
                        *train_command,
                    ],
                    "post_training_commands": [
                        [
                            "python3",
                            METRICS_EXTRACTOR,
                            "--metrics-dir",
                            str(root / "metrics"),
                            "--reward-journal",
                            str(root / "reward_events.jsonl"),
                            "--output",
                            str(root / "metrics_summary.json"),
                        ],
                        [
                            "python3",
                            DIRECTORY_MANIFEST,
                            "--directory",
                            str(root / "metrics"),
                            "--output",
                            str(root / "metrics_manifest.json"),
                        ],
                        [
                            "python3",
                            CHECKPOINT_MANIFEST,
                            "--checkpoint-dir",
                            str(checkpoint),
                            "--output",
                            str(root / "checkpoint_manifest.json"),
                        ],
                    ],
                    "reload_serve_command_template": [
                        "areno",
                        "serve",
                        "--model-path",
                        str(checkpoint),
                        "--tp-size",
                        "1",
                        "--world-size",
                        "1",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "8000",
                        "--max-running-prompts",
                        "1",
                        "--default-max-tokens",
                        "128",
                        "--eager-decode",
                        "--attn-backend",
                        "native",
                        "--disable-thinking",
                    ],
                    "monitored_reload_serve_command_template": [
                        "python3",
                        MONITOR,
                        "--gpu-uuid",
                        str(model["gpu_pairing"]),
                        "--output",
                        str(root / "reload_gpu_monitor.json"),
                        "--log",
                        str(root / "reload_serve.log"),
                        "--",
                        "areno",
                        "serve",
                        "--model-path",
                        str(checkpoint),
                        "--tp-size",
                        "1",
                        "--world-size",
                        "1",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "8000",
                        "--max-running-prompts",
                        "1",
                        "--default-max-tokens",
                        "128",
                        "--eager-decode",
                        "--attn-backend",
                        "native",
                        "--disable-thinking",
                    ],
                    "reload_client_command_template": [
                        "python3",
                        SERVING_CLIENT,
                        "--manifest",
                        "{E1_MANIFEST}",
                        "--family",
                        family,
                        "--runtime-identity",
                        f"{{{family.upper()}_RELOAD_RUNTIME_IDENTITY_JSON}}",
                        "--task-limit",
                        "1",
                        "--base-url",
                        "http://127.0.0.1:8000/v1",
                        "--journal",
                        str(root / "reload_raw_responses.jsonl"),
                        "--output",
                        str(root / "reload_result.json"),
                    ],
                    "required_environment": {
                        "RIST_REQUIRE_EVIDENCE_JOURNALS": "1",
                        "RIST_RAW_JOURNAL_PATH": str(run_root / run_id / "raw_responses.jsonl"),
                        "RIST_REWARD_JOURNAL_PATH": str(run_root / run_id / "reward_events.jsonl"),
                    },
                    "artifact_binding_contract": {
                        "metrics_dir": str(root / "metrics"),
                        "checkpoint_dir": str(checkpoint),
                        "train_log": str(root / "train.log"),
                        "reload_gpu_monitor": str(root / "reload_gpu_monitor.json"),
                        "reload_serve_log": str(root / "reload_serve.log"),
                        "reload_runtime_identity_required_fields": [
                            "parent_run_id",
                            "checkpoint_manifest_sha256",
                            "loaded_checkpoint_path",
                        ],
                    },
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
        "protocol": "RIST-E1-CAPACITY-v2.1",
        "source_commit": source_commit,
        "prerequisite": "PASS_C0_TO_FILTERED_TRAIN",
        "training_seed": TRAINING_SEED,
        "models": MODELS,
        "algorithms": list(ALGORITHMS),
        "capacity_arm": "AF",
        "serving_canary_count": len(serving_canaries),
        "serving_canaries": serving_canaries,
        "job_count": len(jobs),
        "execution_authorized": False,
        "commands_are_templates_only": True,
        "gpu_training_authorized": False,
        "heldout_permitted": False,
        "bfcl_permitted": False,
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.run_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

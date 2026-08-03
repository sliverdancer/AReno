"""Build the outcome-blind M0 third-checkpoint selection freeze from local source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[6]
AUDITED_FILES = (
    "areno/models/__init__.py",
    "areno/models/qwen3/model.py",
    "areno/models/qwen3/checkpoint.py",
    "pyproject.toml",
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/M0_THIRD_CHECKPOINT/PROTOCOL.md",
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/M0_THIRD_CHECKPOINT/build_selection_freeze.py",
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/M0_THIRD_CHECKPOINT/verify_selection_freeze.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    registration = (REPO_ROOT / "areno/models/__init__.py").read_text()
    if "register_adapter(Qwen3Adapter())" not in registration:
        raise ValueError("frozen source does not register the dense Qwen3 adapter")
    qwen_source = (REPO_ROOT / "areno/models/qwen3/model.py").read_text()
    if "class Qwen3Adapter" not in qwen_source:
        raise ValueError("frozen source lacks the dense Qwen3 adapter implementation")
    dependency_manifest = (REPO_ROOT / "pyproject.toml").read_text()
    if "transformers" not in dependency_manifest or "safetensors" not in dependency_manifest:
        raise ValueError("existing checkpoint dependencies are unavailable")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema_version": "rist-m0-third-checkpoint-selection-v1",
        "source_commit": source_commit,
        "main_conference_requirement": "AT_LEAST_3_CHECKPOINTS_ACROSS_2_FAMILIES",
        "existing_checkpoints": [
            {"checkpoint": "Qwen/Qwen3-0.6B", "family": "qwen3"},
            {"checkpoint": "google/gemma-4-E2B-it", "family": "gemma4"},
        ],
        "selection_rule": [
            "EXISTING_REGISTERED_ADAPTER_AND_CHECKPOINT_IO",
            "NO_NEW_RUNTIME_OR_PYTHON_DEPENDENCY",
            "PUBLIC_TEXT_ONLY_DENSE_CAUSAL_LM",
            "PRESERVE_TWO_FAMILIES_AND_ADD_WITHIN_QWEN_SCALE_REPLICATION",
            "SMALLEST_QWEN3_DENSE_SCALE_STRICTLY_ABOVE_0_6B",
            "NO_MODEL_OR_TASK_OUTCOME_INPUT",
        ],
        "primary_candidate": {
            "checkpoint": "Qwen/Qwen3-1.7B",
            "family": "qwen3",
            "exact_revision": None,
            "status": "PROVISIONAL_NOT_DOWNLOADED_NOT_QUALIFIED",
        },
        "pre_scientific_fallback": {
            "checkpoint": "Qwen/Qwen3-4B",
            "family": "qwen3",
            "permitted_only_for": [
                "PRIMARY_METADATA_UNAVAILABLE",
                "PRIMARY_LICENSE_INCOMPATIBLE",
                "PRIMARY_REQUIRES_REMOTE_CODE_OR_NEW_DEPENDENCY",
            ],
            "forbidden_after_any_model_inference_or_scientific_outcome": True,
        },
        "qualification_gates": [
            "M1_TOKENIZER_CONFIG_LICENSE",
            "M2_WEIGHT_IDENTITY",
            "M3_RUNTIME_TREATMENT",
            "M4_C0_RESOLUTION_TRANSPORT",
            "M5_GSPO_GRPO_CAPACITY",
            "M6_POWERED_STEP_AND_TOKEN_MATCHED_REPLICATION",
        ],
        "authorization": {
            "model_download": False,
            "tokenizer_or_config_download": False,
            "inference": False,
            "training": False,
            "heldout_or_bfcl": False,
        },
        "selection_uses_model_outcomes": False,
        "selection_uses_task_outcomes": False,
        "new_dependency_added": False,
        "counts_as_third_checkpoint": False,
        "audited_files": {
            relative: _sha256(REPO_ROOT / relative) for relative in AUDITED_FILES
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

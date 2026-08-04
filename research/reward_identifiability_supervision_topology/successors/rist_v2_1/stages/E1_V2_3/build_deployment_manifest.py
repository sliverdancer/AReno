"""Build E1 v2.3 executable templates only from the exact C0 admission."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from typing import Any

STAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[6]
BASE_BUILDER = STAGE_ROOT.parent / "E1/build_capacity_manifest.py"
AUTHORIZATION = STAGE_ROOT / "STANDING_GPU_AUTHORIZATION_20260804.json"
FAMILIES = ("qwen3", "gemma4")
REQUIRED_IDENTITY_FIELDS = {
    "protocol",
    "family",
    "checkpoint",
    "model_revision",
    "tokenizer_snapshot_sha256",
    "model_weights_sha256",
    "snapshot_verification_sha256",
    "model_path",
    "gpu_name",
    "gpu_uuid",
    "gpu_total_memory_gib",
    "driver_version",
    "cuda_version",
    "torch_version",
    "source_commit",
    "control_commit",
    "runtime_commit",
    "extension_import_sha256",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"E1 v2.3 dependency unavailable: {path.name}")
    spec.loader.exec_module(module)
    return module


def _replace(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for source, target in replacements.items():
            value = value.replace(source, target)
        return value
    if isinstance(value, list):
        return [_replace(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, replacements) for key, item in value.items()}
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _clean_head() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("E1 deployment requires a clean frozen worktree")
    return head


def _validate_authorization() -> dict[str, Any]:
    value = _json(AUTHORIZATION)
    e1 = value.get("e1", {})
    if not (
        value.get("schema_version") == 1
        and value.get("authorized") is True
        and value.get("gate_policy") == "ORDERED_GO_KILL_NO_REPAIR_NO_SELECTIVE_RERUN"
        and value.get("authorization_bypasses_scientific_gates") is False
        and e1.get("families") == ["qwen3", "gemma4"]
        and e1.get("algorithms") == ["gspo", "grpo"]
        and e1.get("optimizer_steps_per_algorithm") == 1
        and e1.get("training") is True
    ):
        raise PermissionError("E1 standing authorization is invalid")
    return value


def _validate_admission(path: Path, source_commit: str) -> dict[str, Any]:
    value = _json(path)
    capacity_path = Path(str(value.get("capacity_train_path", "")))
    resolution_path = Path(str(value.get("resolution_result_path", "")))
    data_manifest_path = Path(str(value.get("capacity_data_manifest_path", "")))
    if not (
        value.get("protocol") == "RIST-E1-v2.3-C0-ADMISSION-v1"
        and value.get("passed") is True
        and value.get("decision") == "PASS_C0_V2_3_GATE_TO_DEPLOYMENT_BOUND_E1"
        and value.get("source_commit") == source_commit
        and value.get("capacity_task_count") == 4
        and value.get("capacity_selection_rule")
        == "LEXICOGRAPHIC_FIRST_TRANSPORTED_HIGH_CELL"
        and value.get("selection_uses_individual_outcomes") is False
        and value.get("execution_authorized") is False
        and value.get("training_permitted") is False
        and value.get("heldout_permitted") is False
        and value.get("bfcl_permitted") is False
        and capacity_path.is_file()
        and _sha256(capacity_path) == value.get("capacity_train_sha256")
        and resolution_path.is_file()
        and _sha256(resolution_path) == value.get("resolution_result_sha256")
        and data_manifest_path.is_file()
        and _sha256(data_manifest_path) == value.get("capacity_data_manifest_sha256")
    ):
        raise PermissionError("E1 deployment requires the exact v2.3 C0 admission")
    return value


def _validate_identity(
    family: str, identity: dict[str, Any], admission: dict[str, Any], source_commit: str
) -> None:
    model_path = Path(str(identity.get("model_path", "")))
    if not (
        set(identity) >= REQUIRED_IDENTITY_FIELDS
        and identity.get("protocol") == "RIST-E1-v2.3-RUNTIME-IDENTITY-v1"
        and identity.get("family") == family
        and identity.get("checkpoint") == admission["models"][family]
        and identity.get("model_revision") == admission["model_revisions"][family]
        and identity.get("tokenizer_snapshot_sha256")
        == admission["tokenizer_snapshot_sha256"][family]
        and identity.get("gpu_uuid") == admission["gpu_uuid"]
        and float(identity.get("gpu_total_memory_gib", 0.0)) >= 79.0
        and identity.get("source_commit") == source_commit
        and identity.get("control_commit") == source_commit
        and identity.get("runtime_commit") == source_commit
        and model_path.is_dir()
        and "{" not in str(model_path)
        and all(
            isinstance(identity.get(field), str) and bool(identity[field])
            for field in ("gpu_name", "driver_version", "cuda_version", "torch_version")
        )
        and all(
            _is_sha256(identity.get(field))
            for field in (
                "tokenizer_snapshot_sha256",
                "model_weights_sha256",
                "snapshot_verification_sha256",
                "extension_import_sha256",
            )
        )
    ):
        raise ValueError(f"{family} runtime identity is not bound to E1 v2.3")


def build_manifest(
    *,
    admission_path: Path,
    runtime_identity_paths: dict[str, Path],
    run_root: Path,
    manifest_output_path: Path,
    source_commit: str | None = None,
) -> dict[str, Any]:
    commit = source_commit if source_commit is not None else _clean_head()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("E1 deployment source must be a full commit")
    _validate_authorization()
    admission = _validate_admission(admission_path, commit)
    if set(runtime_identity_paths) != set(FAMILIES):
        raise ValueError("E1 deployment requires both family runtime identities")
    identities = {family: _json(runtime_identity_paths[family]) for family in FAMILIES}
    for family in FAMILIES:
        _validate_identity(family, identities[family], admission, commit)
    if len({identities[family]["gpu_uuid"] for family in FAMILIES}) != 1:
        raise ValueError("E1 families must use the same bound GPU")

    base = _load("rist_e1_v2_3_base", BASE_BUILDER).build_manifest(run_root)
    replacements = {
        "{E1_HIGH_RESOLUTION_CANARY_JSONL}": str(admission["capacity_train_path"]),
        "{QWEN3_MODEL_PATH}": str(identities["qwen3"]["model_path"]),
        "{GEMMA4_MODEL_PATH}": str(identities["gemma4"]["model_path"]),
        "{QWEN3_GPU_UUID}": str(identities["qwen3"]["gpu_uuid"]),
        "{GEMMA4_LARGER_MEMORY_GPU_UUID}": str(identities["gemma4"]["gpu_uuid"]),
        "{E1_MANIFEST}": str(manifest_output_path),
        "{QWEN3_RUNTIME_IDENTITY_JSON}": str(runtime_identity_paths["qwen3"]),
        "{GEMMA4_RUNTIME_IDENTITY_JSON}": str(runtime_identity_paths["gemma4"]),
        "{QWEN3_RELOAD_RUNTIME_IDENTITY_JSON}": str(run_root / "qwen3-reload-identity.json"),
        "{GEMMA4_RELOAD_RUNTIME_IDENTITY_JSON}": str(run_root / "gemma4-reload-identity.json"),
    }
    manifest = _replace(base, replacements)
    unresolved = sorted(
        set(re.findall(r"\{[A-Z0-9_]+\}|__[A-Z0-9_]+__", json.dumps(manifest, sort_keys=True)))
    )
    if unresolved:
        raise ValueError(f"E1 deployment has unresolved tokens: {unresolved}")
    manifest.update({
        "protocol": "RIST-E1-CAPACITY-v2.1",
        "successor_protocol": "RIST-E1-v2.3-DEPLOYMENT-BOUND-v1",
        "source_commit": commit,
        "prerequisite": "PASS_C0_V2_3_GATE_TO_DEPLOYMENT_BOUND_E1",
        "c0_admission_sha256": _sha256(admission_path),
        "capacity_train_sha256": admission["capacity_train_sha256"],
        "standing_authorization_sha256": _sha256(AUTHORIZATION),
        "runtime_identity_sha256": {
            family: _sha256(runtime_identity_paths[family]) for family in FAMILIES
        },
        "runtime_identities": identities,
        "execution_authorized": True,
        "commands_are_templates_only": False,
        "gpu_training_authorized": True,
        "heldout_permitted": False,
        "bfcl_permitted": False,
    })
    manifest["models"] = {
        family: {
            **manifest["models"][family],
            "model_path": identities[family]["model_path"],
            "gpu_pairing": identities[family]["gpu_uuid"],
            "deployment_gpu_memory_floor_gib": 79,
        }
        for family in FAMILIES
    }
    for row in manifest["serving_canaries"] + manifest["jobs"]:
        row["execution_authorized"] = True
        row["runtime_identity_sha256"] = manifest["runtime_identity_sha256"][row["family"]]
        row["c0_admission_sha256"] = manifest["c0_admission_sha256"]
        row["minimum_gpu_memory_gib"] = 79
    if not (
        manifest["job_count"] == 4
        and {row["algorithm"] for row in manifest["jobs"]} == {"gspo", "grpo"}
        and all(row["max_steps"] == 1 and row["arm"] == "AF" for row in manifest["jobs"])
    ):
        raise ValueError("E1 deployment must remain the four-job one-step AF matrix")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--qwen-runtime-identity", type=Path, required=True)
    parser.add_argument("--gemma-runtime-identity", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("E1 v2.3 deployment manifest must be fresh")
    result = build_manifest(
        admission_path=args.admission,
        runtime_identity_paths={
            "qwen3": args.qwen_runtime_identity,
            "gemma4": args.gemma_runtime_identity,
        },
        run_root=args.run_root,
        manifest_output_path=args.output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

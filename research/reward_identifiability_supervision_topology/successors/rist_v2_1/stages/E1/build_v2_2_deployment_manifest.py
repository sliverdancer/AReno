"""Build an executable E1 v2.2 manifest from sealed C0 and runtime identities."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


STAGE_ROOT = Path(__file__).resolve().parent
BASE_BUILDER = STAGE_ROOT / "build_capacity_manifest.py"
C0_GATE = STAGE_ROOT / "validate_c0_admission.py"
FAMILIES = ("qwen3", "gemma4")
FAMILY_SCOPES = {"qwen3": "E1_QWEN_CAPACITY", "gemma4": "E1_GEMMA_CAPACITY"}


def _sha(path: Path) -> str:
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
        raise RuntimeError(f"E1 dependency is unavailable: {path.name}")
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


def _validate_runtime_identity(
    family: str,
    identity: dict[str, Any],
    gate: dict[str, Any],
) -> None:
    model_path = Path(str(identity.get("model_path", "")))
    if not (
        identity.get("family") == family
        and identity.get("checkpoint") == gate["models"][family]
        and identity.get("model_revision") == gate["model_revisions"][family]
        and identity.get("tokenizer_snapshot_sha256") == gate["tokenizer_snapshot_sha256"][family]
        and identity.get("gpu_uuid") == gate["collection_gpu_uuid"]
        and type(identity.get("gpu_total_memory_gib")) in (int, float)
        and float(identity["gpu_total_memory_gib"]) >= 48.0
        and model_path.is_dir()
        and "{" not in str(model_path)
        and len(str(identity.get("weights_manifest_sha256", ""))) == 64
        and len(str(identity.get("extension_import_sha256", ""))) == 64
    ):
        raise ValueError(f"{family} runtime identity is not deployment-bound to C0")


def _validate_e1_authorization(path: Path) -> dict[str, Any]:
    authorization = _json(path)
    scopes = authorization.get("authorized_scopes", {})
    valid = authorization.get("schema_version") == 1 and authorization.get("authorized") is True
    for family in FAMILIES:
        scope = scopes.get(FAMILY_SCOPES[family], {})
        valid = valid and (
            scope.get("algorithms") == ["gspo", "grpo"]
            and scope.get("optimizer_steps_per_algorithm") == 1
            and scope.get("training") is True
        )
    if not valid or not (
        "P3_48_RUN_DIAGNOSTIC_TRAINING" in authorization.get("not_authorized", [])
        and "HELDOUT_OR_BFCL_ACCESS" in authorization.get("not_authorized", [])
    ):
        raise PermissionError("E1 deployment lacks the exact frozen training authorization")
    return authorization


def build_deployment_manifest(
    *,
    resolution_root: Path,
    authorization_path: Path,
    runtime_identity_paths: dict[str, Path],
    run_root: Path,
) -> dict[str, Any]:
    gate_module = _load("rist_e1_v2_2_c0_gate", C0_GATE)
    gate = gate_module.validate_c0_admission(resolution_root)
    _validate_e1_authorization(authorization_path)
    identities = {family: _json(runtime_identity_paths[family]) for family in FAMILIES}
    for family in FAMILIES:
        _validate_runtime_identity(family, identities[family], gate)

    base_module = _load("rist_e1_v2_2_base_manifest", BASE_BUILDER)
    base = base_module.build_manifest(run_root)
    capacity_path = Path(gate["capacity_train_path"])
    if _sha(capacity_path) != gate["capacity_train_sha256"]:
        raise ValueError("C0 capacity dataset changed after E1 gate")
    replacements = {
        "{E1_HIGH_RESOLUTION_CANARY_JSONL}": str(capacity_path),
        "{QWEN3_MODEL_PATH}": str(identities["qwen3"]["model_path"]),
        "{GEMMA4_MODEL_PATH}": str(identities["gemma4"]["model_path"]),
        "{QWEN3_GPU_UUID}": str(identities["qwen3"]["gpu_uuid"]),
        "{GEMMA4_LARGER_MEMORY_GPU_UUID}": str(identities["gemma4"]["gpu_uuid"]),
    }
    manifest = _replace(base, replacements)
    unresolved = json.dumps(manifest, sort_keys=True)
    if any(token in unresolved for token in replacements) or "{E1_" in unresolved:
        raise ValueError("E1 deployment manifest contains an unresolved execution token")

    validation_path = resolution_root / "VALIDATION_RESULT.json"
    admission_path = resolution_root / "e1/E1_ADMISSION.json"
    final_path = resolution_root / "FINAL_RESULT.json"
    manifest.update(
        {
            "protocol": "RIST-E1-CAPACITY-v2.2-DEPLOYMENT-BOUND",
            "prerequisite": "PASS_C0_GATE_TO_DEPLOYMENT_BOUND_E1",
            "c0_gate": gate,
            "c0_validation_sha256": _sha(validation_path),
            "c0_admission_sha256": _sha(admission_path),
            "c0_final_sha256": _sha(final_path),
            "capacity_train_sha256": gate["capacity_train_sha256"],
            "external_authorization_sha256": _sha(authorization_path),
            "runtime_identity_sha256": {
                family: _sha(runtime_identity_paths[family]) for family in FAMILIES
            },
            "runtime_identities": identities,
            "execution_authorized": True,
            "commands_are_templates_only": False,
            "gpu_training_authorized": True,
            "heldout_permitted": False,
            "bfcl_permitted": False,
        }
    )
    for row in manifest["serving_canaries"] + manifest["jobs"]:
        row["execution_authorized"] = True
        family = row["family"]
        row["c0_validation_sha256"] = manifest["c0_validation_sha256"]
        row["runtime_identity_sha256"] = manifest["runtime_identity_sha256"][family]
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--qwen-runtime-identity", type=Path, required=True)
    parser.add_argument("--gemma-runtime-identity", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_deployment_manifest(
        resolution_root=args.resolution_root,
        authorization_path=args.authorization,
        runtime_identity_paths={
            "qwen3": args.qwen_runtime_identity,
            "gemma4": args.gemma_runtime_identity,
        },
        run_root=args.run_root,
    )
    if args.output.exists():
        raise FileExistsError("E1 deployment manifest output must be fresh")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

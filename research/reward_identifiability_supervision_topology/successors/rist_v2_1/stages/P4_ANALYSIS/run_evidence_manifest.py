"""Build and validate one complete P4 run-evidence manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CHECKPOINT_STEPS = (25, 50, 75, 100)
REQUIRED_ROLES = {
    "source_archive",
    "runtime_identity",
    "resolution_map",
    "training_raw_journal",
    "training_reward_journal",
    "training_metrics_manifest",
    "training_summary",
    "confirmatory_result",
    "confirmatory_journal",
    "confirmatory_ledger",
    *(f"checkpoint_manifest_{step:03d}" for step in CHECKPOINT_STEPS),
    *(f"dev_result_{step:03d}" for step in CHECKPOINT_STEPS),
    *(f"dev_journal_{step:03d}" for step in CHECKPOINT_STEPS),
}


def _safe_file(root: Path, relative: str) -> Path:
    path_value = Path(relative)
    if not path_value.parts or path_value.is_absolute() or ".." in path_value.parts:
        raise ValueError(f"unsafe evidence path: {relative!r}")
    root = root.resolve()
    path = (root / path_value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"evidence artifact is missing: {relative}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(run_id: str, evidence_root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    if binding.get("protocol") != "RIST-P4-RUN-EVIDENCE-BINDING-v1":
        raise ValueError("unexpected run-evidence binding")
    if binding.get("run_id") != run_id:
        raise ValueError("run-evidence binding belongs to another run")
    paths = binding.get("artifacts")
    if not isinstance(paths, dict) or set(paths) != REQUIRED_ROLES:
        raise ValueError("run-evidence binding does not contain the exact required roles")
    artifacts = []
    seen_paths = set()
    for role in sorted(REQUIRED_ROLES):
        relative = paths[role]
        if not isinstance(relative, str) or relative in seen_paths:
            raise ValueError("run-evidence paths must be unique strings")
        seen_paths.add(relative)
        path = _safe_file(evidence_root, relative)
        artifacts.append(
            {
                "role": role,
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "protocol": "RIST-P4-RUN-EVIDENCE-MANIFEST-v1",
        "run_id": run_id,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def validate_manifest(
    manifest: dict[str, Any], evidence_root: Path, expected_run_id: str
) -> dict[str, Any]:
    if manifest.get("protocol") != "RIST-P4-RUN-EVIDENCE-MANIFEST-v1":
        raise ValueError("unexpected run-evidence manifest")
    if manifest.get("run_id") != expected_run_id:
        raise ValueError("run-evidence manifest belongs to another run")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or manifest.get("artifact_count") != len(rows):
        raise ValueError("run-evidence artifact count mismatch")
    by_role = {}
    seen_paths = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("role"), str):
            raise ValueError("invalid run-evidence artifact row")
        role = row["role"]
        relative = row.get("path")
        if role in by_role or not isinstance(relative, str) or relative in seen_paths:
            raise ValueError("duplicate run-evidence role or path")
        seen_paths.add(relative)
        path = _safe_file(evidence_root, relative)
        if row.get("size") != path.stat().st_size or row.get("sha256") != _sha256(path):
            raise ValueError(f"run-evidence artifact changed: {role}")
        by_role[role] = row
    if set(by_role) != REQUIRED_ROLES:
        raise ValueError("run-evidence manifest does not contain the exact required roles")
    return {
        "protocol": "RIST-P4-RUN-EVIDENCE-VERIFY-v1",
        "run_id": expected_run_id,
        "artifact_count": len(rows),
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--run-id", required=True)
    build.add_argument("--evidence-root", type=Path, required=True)
    build.add_argument("--binding", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--evidence-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_manifest(
            args.run_id, args.evidence_root, json.loads(args.binding.read_text())
        )
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        result = validate_manifest(
            json.loads(args.manifest.read_text()), args.evidence_root, args.run_id
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

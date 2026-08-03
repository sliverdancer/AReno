"""Verify every file hash and authorization hash in the GPU execution freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def verify(freeze_path: Path) -> dict[str, Any]:
    base = freeze_path.parent
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("schema_version") != "rist-gpu-execution-freeze-v2":
        raise ValueError("unexpected GPU execution freeze")
    authorization = base / "GPU_AUTHORIZATION_20260803.json"
    authorization_pass = (
        hashlib.sha256(authorization.read_bytes()).hexdigest()
        == freeze["authorization_sha256"]
    )
    files = {}
    for relative, expected in freeze["files"].items():
        path = base / relative
        files[relative] = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    source_bound_files = (
        "stages/C0_RESOLUTION_V2_2/CANARY_EXECUTION_MANIFEST.json",
        "stages/C0_RESOLUTION_V2_2/CANARY_CPU_FREEZE.json",
        "stages/E1/EXECUTION_MANIFEST.json",
        "stages/E1/CPU_FREEZE.json",
    )
    stage_source_commits = {
        relative: json.loads((base / relative).read_text()).get("source_commit")
        for relative in source_bound_files
    }
    source_binding_pass = all(
        value == freeze["source_commit"] for value in stage_source_commits.values()
    )
    passed = authorization_pass and all(files.values()) and source_binding_pass
    return {
        "protocol": "RIST-GPU-EXECUTION-FREEZE-VERIFY-v2",
        "source_commit": freeze["source_commit"],
        "authorization_pass": authorization_pass,
        "file_count": len(files),
        "files": files,
        "stage_source_commits": stage_source_commits,
        "source_binding_pass": source_binding_pass,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.freeze)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

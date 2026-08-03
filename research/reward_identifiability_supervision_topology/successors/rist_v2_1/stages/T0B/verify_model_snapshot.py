"""Verify an exact, minimal T0b model snapshot against its frozen lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()


def verify_snapshot(root: Path, model_cell: str, lock: dict[str, Any]) -> dict[str, Any]:
    models = lock.get("models")
    if not isinstance(models, dict) or model_cell not in models:
        raise ValueError("unknown frozen model cell")
    if not root.is_dir():
        raise ValueError("model snapshot root is missing")
    model = models[model_cell]
    expected = {str(row["path"]): row for row in model["files"]}
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }
    if set(actual) != set(expected):
        raise ValueError("model snapshot file set differs from frozen lock")
    rows = []
    for relative in sorted(expected):
        path = actual[relative]
        if path.is_symlink():
            raise ValueError(f"model snapshot may not contain symlinks: {relative}")
        payload = path.read_bytes()
        spec = expected[relative]
        if len(payload) != int(spec["size"]):
            raise ValueError(f"model snapshot size mismatch: {relative}")
        if spec["digest_kind"] == "sha256":
            digest = hashlib.sha256(payload).hexdigest()
        elif spec["digest_kind"] == "git_blob_sha1":
            digest = _git_blob_sha1(payload)
        else:
            raise ValueError("unsupported frozen digest kind")
        if digest != spec["digest"]:
            raise ValueError(f"model snapshot digest mismatch: {relative}")
        rows.append({"path": relative, "size": len(payload), "digest": digest})
    return {
        "protocol": lock["protocol"],
        "model_cell": model_cell,
        "repo_id": model["repo_id"],
        "revision": model["revision"],
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "files": rows,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-cell", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    result = verify_snapshot(args.root, args.model_cell, lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

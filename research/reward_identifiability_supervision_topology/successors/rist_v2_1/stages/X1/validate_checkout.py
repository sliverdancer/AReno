"""Fail-closed validation for a future pinned Tau3 or BFCL checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _normalized_remote(value: str) -> str:
    return value.strip().rstrip("/").removesuffix(".git").lower()


def validate_checkout(
    root: Path, source_name: str, source_lock: dict[str, Any]
) -> dict[str, Any]:
    """Validate identity and license without reading benchmark task content."""

    if source_name not in {"tau3", "bfcl"}:
        raise ValueError("source_name must be tau3 or bfcl")
    expected = source_lock[source_name]
    head = _git(root, "rev-parse", "HEAD")
    if head != expected["commit"]:
        raise ValueError(f"commit mismatch for {source_name}")
    tag_object = _git(root, "rev-parse", str(expected["tag"]))
    if tag_object != expected["tag_object"]:
        raise ValueError(f"tag object mismatch for {source_name}")
    peeled_commit = _git(root, "rev-parse", f"{expected['tag']}^{{}}")
    if peeled_commit != expected["commit"]:
        raise ValueError(f"tag does not peel to frozen commit for {source_name}")
    if _git(root, "status", "--porcelain"):
        raise ValueError(f"checkout must be clean for {source_name}")
    remote = _git(root, "remote", "get-url", "origin")
    if _normalized_remote(remote) != _normalized_remote(str(expected["repository"])):
        raise ValueError(f"origin mismatch for {source_name}")

    license_candidates = [
        path
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")
        if (path := root / name).is_file()
    ]
    if len(license_candidates) != 1:
        raise ValueError(f"expected exactly one root license file for {source_name}")
    license_path = license_candidates[0]
    license_text = license_path.read_text(encoding="utf-8", errors="strict")
    expected_license = str(expected["expected_license"])
    license_markers = {
        "MIT": "MIT License",
        "Apache-2.0": "Apache License",
    }
    marker = license_markers[expected_license]
    if marker.lower() not in license_text.lower():
        raise ValueError(f"license marker mismatch for {source_name}")

    return {
        "source": source_name,
        "commit": head,
        "tag": expected["tag"],
        "tag_object": tag_object,
        "peeled_commit": peeled_commit,
        "origin": remote,
        "clean": True,
        "license": expected_license,
        "license_path": license_path.name,
        "license_sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
        "benchmark_content_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", choices=("tau3", "bfcl"), required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    result = validate_checkout(args.root, args.source, lock)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

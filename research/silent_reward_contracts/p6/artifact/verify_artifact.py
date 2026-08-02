"""Verify the extracted ARCA anonymous artifact using only the standard library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any


MANIFEST_NAME = "artifact_manifest.json"
FORBIDDEN_PATH_PARTS = {".git", "__pycache__", ".pytest_cache"}
FORBIDDEN_TEXT = (
    ("local Linux path", re.compile(r"(?<![A-Za-z0-9])/(?:mnt|home)/[^\s'\"}]+")),
    ("local Windows path", re.compile(r"\b[A-Za-z]:\\\\(?:Users|home)\\\\")),
    ("SSH endpoint", re.compile(r"\bssh\s+(?:-[^\n]+\s+)?[^\s@]+@[^\s]+", re.IGNORECASE)),
    ("email address", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("private key", re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY")),
    ("access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("API token assignment", re.compile(r"\b(?:api[_-]?key|token|password)\s*=\s*['\"][^'\"]+", re.IGNORECASE)),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scan_text(relative_path: str, text: str) -> list[str]:
    """Return anonymity or secret-pattern violations for one UTF-8 file."""

    violations: list[str] = []
    for label, pattern in FORBIDDEN_TEXT:
        if pattern.search(text):
            violations.append(f"{relative_path}: {label}")
    return violations


def load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "arca.anonymous-artifact.v1":
        raise ValueError("unexpected artifact schema")
    if manifest.get("protocol_id") != "ARCA-P6-PAPER-v0.1":
        raise ValueError("unexpected protocol id")
    return manifest


def verify_directory(root: Path) -> dict[str, Any]:
    """Fail closed on an unexpected, modified, malformed, or identifying file."""

    root = root.resolve()
    manifest = load_manifest(root)
    expected = {entry["path"]: entry for entry in manifest["files"]}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        raise ValueError(f"artifact allowlist mismatch: missing={missing}, unexpected={unexpected}")

    violations: list[str] = []
    for relative_path, entry in sorted(expected.items()):
        path = root / relative_path
        if any(part in FORBIDDEN_PATH_PARTS for part in path.parts):
            violations.append(f"{relative_path}: forbidden path component")
        data = path.read_bytes()
        if len(data) != entry["size"]:
            violations.append(f"{relative_path}: size mismatch")
        if sha256_bytes(data) != entry["sha256"]:
            violations.append(f"{relative_path}: SHA-256 mismatch")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        violations.extend(scan_text(relative_path, text))
        if relative_path.endswith(".json"):
            json.loads(text)
        elif relative_path.endswith(".csv"):
            rows = list(csv.reader(text.splitlines()))
            if not rows or not rows[0] or any(not cell.strip() for cell in rows[0]):
                violations.append(f"{relative_path}: missing CSV header")

    if violations:
        raise ValueError("artifact audit failed: " + "; ".join(violations))
    return {
        "protocol_id": manifest["protocol_id"],
        "file_count": len(expected),
        "payload_sha256": manifest["payload_sha256"],
        "status": "PASS_ANONYMOUS_ARTIFACT_AUDIT",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args()
    print(json.dumps(verify_directory(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

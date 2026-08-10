"""Write SHA-256 manifest for the negative-result artifact package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


INCLUDE_SUFFIXES = {".md", ".csv", ".json", ".py", ".svg"}
EXCLUDE_NAMES = {"ARTIFACT_MANIFEST.json"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    files = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.suffix in INCLUDE_SUFFIXES
            and path.name not in EXCLUDE_NAMES
            and "__pycache__" not in path.parts
        ):
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "protocol": "RIST-NEGATIVE-RESULT-ARTIFACT-MANIFEST-v1",
        "root": root.as_posix(),
        "file_count": len(files),
        "files": files,
        "sealed_or_heldout_data_included": False,
        "model_weights_included": False,
        "gpu_required_to_reproduce_tables": False,
    }
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

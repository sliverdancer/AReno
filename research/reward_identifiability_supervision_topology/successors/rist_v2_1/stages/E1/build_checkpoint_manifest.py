"""Hash every file in one saved E1 checkpoint for reload verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_manifest(checkpoint_dir: Path) -> dict:
    files = []
    for path in sorted(value for value in checkpoint_dir.rglob("*") if value.is_file()):
        files.append(
            {
                "path": path.relative_to(checkpoint_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not files or not any(row["path"].endswith(".safetensors") for row in files):
        raise ValueError("E1 checkpoint is missing safetensors weights")
    return {
        "protocol": "RIST-E1-CHECKPOINT-MANIFEST-v1",
        "file_count": len(files),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.checkpoint_dir)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

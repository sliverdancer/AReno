"""Hash every regular file under a directory without archiving or mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_manifest(directory: Path) -> dict:
    files = [
        {
            "path": path.relative_to(directory).as_posix(),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    if not files:
        raise ValueError("directory manifest cannot be empty")
    return {"protocol": "RIST-DIRECTORY-MANIFEST-v1", "file_count": len(files), "files": files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(build_manifest(args.directory), indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download or verify the frozen ModelScope asset for the CARe P3 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(__file__).with_name("modelscope_asset.json")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate the frozen asset manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("hub") != "modelscope":
        raise ValueError("asset manifest hub must be modelscope")
    if not payload.get("model_id") or not payload.get("revision"):
        raise ValueError("asset manifest requires model_id and revision")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("asset manifest requires a non-empty files list")
    seen: set[str] = set()
    for entry in files:
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe asset path: {relative}")
        if str(relative) in seen:
            raise ValueError(f"duplicate asset path: {relative}")
        seen.add(str(relative))
        if int(entry["size"]) < 0 or len(str(entry["sha256"])) != 64:
            raise ValueError(f"invalid asset metadata for {relative}")
    return payload


def verify_snapshot(
    local_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify every frozen file by byte length and SHA-256."""

    failures = []
    verified = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = local_dir / relative
        if not path.is_file():
            failures.append({"path": str(relative), "reason": "missing"})
            continue
        observed_size = path.stat().st_size
        observed_sha256 = _sha256(path)
        expected_size = int(entry["size"])
        expected_sha256 = str(entry["sha256"])
        if observed_size != expected_size or observed_sha256 != expected_sha256:
            failures.append(
                {
                    "path": str(relative),
                    "reason": "content_mismatch",
                    "expected_size": expected_size,
                    "observed_size": observed_size,
                    "expected_sha256": expected_sha256,
                    "observed_sha256": observed_sha256,
                }
            )
            continue
        verified.append(str(relative))
    if failures:
        raise ValueError(
            "ModelScope snapshot does not match the frozen asset manifest: "
            + json.dumps(failures, sort_keys=True)
        )
    return {
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "local_dir": str(local_dir.resolve()),
        "verified_file_count": len(verified),
        "verified_files": verified,
    }


def download_snapshot(
    local_dir: Path,
    manifest: dict[str, Any],
) -> Path:
    """Download the declared ModelScope branch into an explicit directory."""

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "ModelScope download requires the modelscope package."
        ) from exc
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    resolved = Path(
        snapshot_download(
            model_id=str(manifest["model_id"]),
            revision=str(manifest["revision"]),
            local_dir=str(local_dir),
        )
    )
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download before verification. Without this flag, verify only.",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest.expanduser().resolve())
    local_dir = args.local_dir.expanduser().resolve()
    if args.download:
        resolved = download_snapshot(local_dir, manifest)
        if resolved.resolve() != local_dir:
            raise ValueError(
                f"ModelScope returned unexpected snapshot path: {resolved}"
            )
    summary = verify_snapshot(local_dir, manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

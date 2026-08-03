"""Authorized-only downloader for a frozen T0b model snapshot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path


def _load_verifier():
    path = Path(__file__).with_name("verify_model_snapshot.py")
    spec = importlib.util.spec_from_file_location("rist_t0b_model_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen model verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download_snapshot(model_cell: str, output_dir: Path, lock: dict, *, download_file):
    if lock.get("download_permitted") is not True:
        raise PermissionError("frozen lock does not authorize model download")
    if output_dir.exists():
        raise FileExistsError("refusing to replace an existing model snapshot")
    model = lock["models"][model_cell]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        for row in model["files"]:
            cached = Path(
                download_file(
                    model["repo_id"],
                    filename=row["path"],
                    revision=model["revision"],
                )
            )
            (staging / row["path"]).write_bytes(cached.read_bytes())
        result = _load_verifier().verify_snapshot(staging, model_cell, lock)
        os.replace(staging, output_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cell", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("authorized model download requires huggingface_hub") from exc
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    result = download_snapshot(
        args.model_cell,
        args.output_dir,
        lock,
        download_file=hf_hub_download,
    )
    args.verification_output.parent.mkdir(parents=True, exist_ok=True)
    args.verification_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

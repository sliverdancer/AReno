"""Download a revision-pinned, tokenizer-only Hugging Face snapshot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Callable



def _load_capture_module():
    path = Path(__file__).with_name("capture_mask_fixture.py")
    spec = importlib.util.spec_from_file_location("rist_v2_1_t0_capture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen tokenizer snapshot validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CAPTURE = _load_capture_module()
_SAFE_TOKENIZER_FILES = _CAPTURE._SAFE_TOKENIZER_FILES
tokenizer_snapshot_manifest = _CAPTURE.tokenizer_snapshot_manifest


def select_allowlisted_files(repo_files: list[str]) -> list[str]:
    """Select only root-level files accepted by the frozen snapshot validator."""

    selected = sorted(
        filename
        for filename in repo_files
        if "/" not in filename and filename in _SAFE_TOKENIZER_FILES
    )
    if "tokenizer_config.json" not in selected:
        raise ValueError("remote revision lacks tokenizer_config.json")
    return selected


def download_snapshot(
    repo_id: str,
    revision: str,
    output_dir: Path,
    *,
    list_repo_files: Callable[..., list[str]],
    download_file: Callable[..., str],
) -> dict[str, object]:
    """Materialize an isolated snapshot without weights, symlinks, or cache data."""

    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("revision must be a lowercase 40-character commit SHA")
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing snapshot: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    selected = select_allowlisted_files(list_repo_files(repo_id, revision=revision))
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        for filename in selected:
            cached = Path(
                download_file(repo_id, filename=filename, revision=revision)
            )
            (staging / filename).write_bytes(cached.read_bytes())
        manifest = tokenizer_snapshot_manifest(staging)
        os.replace(staging, output_dir)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "selected_files": selected,
        "snapshot": manifest,
        "model_weights_downloaded": False,
        "trust_remote_code": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("tokenizer download requires huggingface_hub") from exc
    result = download_snapshot(
        args.repo_id,
        args.revision,
        args.output_dir,
        list_repo_files=HfApi().list_repo_files,
        download_file=hf_hub_download,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

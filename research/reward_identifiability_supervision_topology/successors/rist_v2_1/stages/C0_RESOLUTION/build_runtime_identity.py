"""Bind one C0 endpoint to exact source, model snapshot, tokenizer, and GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def build_identity(
    manifest: dict[str, Any],
    family: str,
    snapshot: dict[str, Any],
    snapshot_sha256: str,
    gpu: dict[str, Any],
    endpoint: str,
) -> dict[str, Any]:
    if snapshot.get("passed") is not True:
        raise ValueError("model snapshot verification did not pass")
    if (
        snapshot.get("repo_id") != manifest["models"][family]
        or snapshot.get("revision") != manifest["model_revisions"][family]
    ):
        raise ValueError("model snapshot identity does not match C0 manifest")
    weight_digests = [
        str(row["digest"])
        for row in snapshot.get("files", [])
        if str(row.get("path", "")).endswith(".safetensors")
    ]
    if len(weight_digests) != 1 or len(weight_digests[0]) != 64:
        raise ValueError("C0 currently requires one SHA-256 safetensors weight file")
    for field in ("gpu_name", "gpu_uuid", "driver_version", "cuda_version", "torch_version"):
        if not isinstance(gpu.get(field), str) or not gpu[field]:
            raise ValueError(f"GPU identity requires {field}")
    total_memory = float(gpu.get("gpu_total_memory_gib", 0.0))
    if total_memory <= 0.0:
        raise ValueError("GPU identity requires positive gpu_total_memory_gib")
    if not endpoint.startswith("http://127.0.0.1:"):
        raise ValueError("C0 endpoint must be loopback-only")
    return {
        "protocol": "RIST-C0-RUNTIME-IDENTITY-v1",
        "family": family,
        "checkpoint": manifest["models"][family],
        "model_revision": manifest["model_revisions"][family],
        "tokenizer_snapshot_sha256": manifest["tokenizer_snapshot_sha256"][family],
        "model_weights_sha256": weight_digests[0],
        "snapshot_verification_sha256": snapshot_sha256,
        "source_commit": manifest["source_commit"],
        "gpu_name": gpu["gpu_name"],
        "gpu_uuid": gpu["gpu_uuid"],
        "gpu_total_memory_gib": total_memory,
        "driver_version": gpu["driver_version"],
        "cuda_version": gpu["cuda_version"],
        "torch_version": gpu["torch_version"],
        "endpoint": endpoint,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family", choices=("qwen3", "gemma4"), required=True)
    parser.add_argument("--snapshot-verification", type=Path, required=True)
    parser.add_argument("--gpu-identity", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot_bytes = args.snapshot_verification.read_bytes()
    result = build_identity(
        json.loads(args.manifest.read_text()),
        args.family,
        json.loads(snapshot_bytes),
        hashlib.sha256(snapshot_bytes).hexdigest(),
        json.loads(args.gpu_identity.read_text()),
        args.endpoint,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

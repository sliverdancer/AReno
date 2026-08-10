"""Bind non-circular live runtime identities into a RIST C0 v3 manifest.

This runs on the target host before receipt generation. It probes source commit,
model revisions, GPU UUID, and interpreter identity, but it deliberately does not
write deployment receipt SHA-256 into the manifest. Receipt SHA depends on the
manifest SHA, so pre-binding it would create a circular protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import deployment_entrypoint as gate


def _write_json_fresh(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def static_identity(
    *,
    family: str,
    source_root: Path,
    model_path: Path,
    interpreter_path: Path,
    probes: gate.Probes = gate.DEFAULT_PROBES,
) -> dict[str, str]:
    interpreter = interpreter_path.resolve(strict=True)
    return {
        "family": family,
        "source_commit": probes.git_head(source_root.resolve(strict=True)),
        "model_revision": probes.model_revision(model_path),
        "gpu_uuid": probes.gpu_uuid(),
        "interpreter_realpath": str(interpreter),
        "interpreter_sha256": probes.file_sha256(interpreter),
        "interpreter_version": probes.interpreter_version(interpreter),
    }


def bind_manifest(
    *,
    manifest: Mapping[str, Any],
    source_root: Path,
    model_paths: Mapping[str, Path],
    interpreter_path: Path,
    probes: gate.Probes = gate.DEFAULT_PROBES,
) -> dict[str, Any]:
    if manifest.get("protocol") != "RIST-C0-v3.0-STAGE-MANIFEST-v1":
        raise ValueError("unexpected RIST C0 v3 stage manifest")
    value = json.loads(json.dumps(manifest, sort_keys=True))
    for job in value.get("jobs", []):
        family = str(job.get("family"))
        if family not in model_paths:
            raise ValueError(f"unexpected RIST v3 family: {family}")
        identity = static_identity(
            family=family,
            source_root=source_root,
            model_path=model_paths[family],
            interpreter_path=interpreter_path,
            probes=probes,
        )
        if "deployment_receipt_sha256" in identity:
            raise AssertionError("static identity must not include receipt SHA")
        job["runtime_identity"] = identity
    if {job.get("family") for job in value.get("jobs", [])} != set(model_paths):
        raise ValueError("manifest must contain exactly the supplied model families")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-input", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--qwen-model-path", type=Path, required=True)
    parser.add_argument("--gemma-model-path", type=Path, required=True)
    parser.add_argument("--interpreter", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest_input.read_text(encoding="utf-8"))
    value = bind_manifest(
        manifest=manifest,
        source_root=args.source_root,
        model_paths={"qwen3": args.qwen_model_path, "gemma4": args.gemma_model_path},
        interpreter_path=args.interpreter,
    )
    _write_json_fresh(args.manifest_output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

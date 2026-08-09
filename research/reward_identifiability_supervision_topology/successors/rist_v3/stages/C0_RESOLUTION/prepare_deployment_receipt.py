"""Prepare a RIST C0 v3 deployment authority and receipt on the target host.

This tool performs no serving, model request, inference, training, qualification,
held-out, or BFCL access. It only probes immutable deployment identities and
writes the canonical authority/receipt artifacts consumed by deployment_entrypoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import deployment_entrypoint as gate


def _write_bytes_fresh(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_text_fresh(path: Path, text: str) -> None:
    _write_bytes_fresh(path, text.encode("utf-8"))


def _authority_bytes(authority: Mapping[str, Any]) -> bytes:
    return gate._canonical(dict(authority)) + b"\n"


def prepare_receipt(
    *,
    source_root: Path,
    stage_manifest: Path,
    model_path: Path,
    extension_path: Path,
    interpreter_path: Path,
    launcher_command: Sequence[str],
    probes: gate.Probes = gate.DEFAULT_PROBES,
) -> dict[str, Any]:
    inputs = gate.LiveInputs(source_root, model_path, extension_path, interpreter_path)
    manifest_path = stage_manifest.resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    observed = gate._observed(inputs, manifest_bytes, probes)
    authority = {"protocol": gate.AUTHORITY_PROTOCOL, **observed}
    authority_sha256 = gate._bytes_sha256(gate._canonical(authority))
    receipt = gate.build_receipt(
        inputs=inputs,
        manifest_path=manifest_path,
        launcher_command=launcher_command,
        authority=authority,
        authorized_authority_sha256=authority_sha256,
        probes=probes,
    )
    receipt_bytes = gate.receipt_artifact_bytes(receipt)
    return {
        "protocol": "RIST-C0-v3.0-DEPLOYMENT-PREFLIGHT-v1",
        "authority": authority,
        "authority_sha256": authority_sha256,
        "receipt": receipt,
        "receipt_artifact_sha256": gate._bytes_sha256(receipt_bytes),
        "model_accessed": False,
        "request_sent": False,
        "serving_started": False,
        "training_performed": False,
        "qualification_accessed": False,
        "heldout_accessed": False,
        "bfcl_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--interpreter", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--launcher-command-json",
        required=True,
        help="JSON list; first element must equal the bound interpreter real path.",
    )
    parser.add_argument("--authority-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--receipt-sha256-output", type=Path, required=True)
    args = parser.parse_args()

    launcher_command = json.loads(args.launcher_command_json)
    if not isinstance(launcher_command, list) or not all(
        isinstance(part, str) for part in launcher_command
    ):
        raise TypeError("launcher-command-json must be a JSON list of strings")
    result = prepare_receipt(
        source_root=args.source_root,
        stage_manifest=args.stage_manifest,
        model_path=args.model_path,
        extension_path=args.extension,
        interpreter_path=args.interpreter,
        launcher_command=launcher_command,
    )
    _write_bytes_fresh(args.authority_output, _authority_bytes(result["authority"]))
    _write_bytes_fresh(args.receipt_output, gate.receipt_artifact_bytes(result["receipt"]))
    _write_text_fresh(
        args.receipt_sha256_output,
        result["receipt_artifact_sha256"] + "  " + args.receipt_output.name + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Single fail-closed deployment entrypoint for RIST C0 v2.3.

The control side creates a self-contained receipt.  At launch, the runtime
receives no manifest path: it verifies the embedded bytes, probes every live
identity, records an append-only decision, and only then crosses the subprocess
boundary.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROTOCOL = "RIST-C0-v2.3-DEPLOYMENT-RECEIPT-v1"
AUTHORITY_PROTOCOL = "RIST-C0-v2.3-PRE-RENT-AUTHORITY-v1"
FIELDS = (
    "control_commit",
    "runtime_commit",
    "manifest_sha256",
    "model_revision",
    "gpu_uuid",
    "extension_sha256",
)


class DeploymentRefused(RuntimeError):
    """Raised before a request-capable subprocess can be started."""


@dataclass(frozen=True)
class LiveInputs:
    control_root: Path
    runtime_root: Path
    model_path: Path
    extension_path: Path


@dataclass(frozen=True)
class Probes:
    git_head: Callable[[Path], str]
    model_revision: Callable[[Path], str]
    gpu_uuid: Callable[[], str]
    file_sha256: Callable[[Path], str]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def receipt_artifact_bytes(receipt: Mapping[str, Any]) -> bytes:
    """Return the one authorized on-disk representation of a receipt."""
    return _canonical(receipt) + b"\n"


def _file_sha256(path: Path) -> str:
    return _bytes_sha256(path.read_bytes())


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _model_revision(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if resolved.parent.name != "snapshots":
        raise DeploymentRefused("model path must resolve to snapshots/<revision>")
    return resolved.name


def _gpu_uuid() -> str:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    values = [line.strip() for line in output.splitlines() if line.strip()]
    if len(values) != 1 or not values[0].startswith("GPU-"):
        raise DeploymentRefused("exactly one concrete GPU UUID is required")
    return values[0]


DEFAULT_PROBES = Probes(_git_head, _model_revision, _gpu_uuid, _file_sha256)


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, (_canonical(dict(event)) + b"\n"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _observed(inputs: LiveInputs, manifest_bytes: bytes, probes: Probes) -> dict[str, str]:
    return {
        "control_commit": probes.git_head(inputs.control_root.resolve()),
        "runtime_commit": probes.git_head(inputs.runtime_root.resolve()),
        "manifest_sha256": _bytes_sha256(manifest_bytes),
        "model_revision": probes.model_revision(inputs.model_path),
        "gpu_uuid": probes.gpu_uuid(),
        "extension_sha256": probes.file_sha256(inputs.extension_path),
    }


def build_receipt(
    *,
    inputs: LiveInputs,
    manifest_path: Path,
    launcher_command: Sequence[str],
    authority: Mapping[str, Any],
    authorized_authority_sha256: str,
    probes: Probes = DEFAULT_PROBES,
) -> dict[str, Any]:
    """Create the sole control artifact; manifest bytes never need reopening."""
    control_root = inputs.control_root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    try:
        manifest_path.relative_to(control_root)
    except ValueError as exc:
        raise ValueError("manifest must be inside the control worktree") from exc
    manifest_bytes = manifest_path.read_bytes()
    json.loads(manifest_bytes)
    if not launcher_command or not all(isinstance(part, str) and part for part in launcher_command):
        raise ValueError("launcher command must be non-empty strings")
    authority_sha256 = _bytes_sha256(_canonical(authority))
    if authority_sha256 != authorized_authority_sha256:
        raise DeploymentRefused("pre-rent authority SHA mismatch")
    if set(authority) != {
        "protocol", "control_commit", "runtime_commit", "manifest_sha256", "model_revisions"
    } or authority.get("protocol") != AUTHORITY_PROTOCOL:
        raise DeploymentRefused("invalid pre-rent authority schema or protocol")
    observed = _observed(inputs, manifest_bytes, probes)
    creation_mismatches = [
        field
        for field in ("control_commit", "runtime_commit", "manifest_sha256")
        if observed[field] != authority[field]
    ]
    revisions = authority["model_revisions"]
    if not isinstance(revisions, list) or observed["model_revision"] not in revisions:
        creation_mismatches.append("model_revision")
    if creation_mismatches:
        raise DeploymentRefused(
            "pre-rent authority mismatch: " + ", ".join(creation_mismatches)
        )
    body = {
        "protocol": PROTOCOL,
        "authority_sha256": authority_sha256,
        "bindings": observed,
        "manifest_base64": base64.b64encode(manifest_bytes).decode("ascii"),
        "launcher_command": list(launcher_command),
    }
    return {**body, "receipt_sha256": _bytes_sha256(_canonical(body))}


def _unpack_receipt(receipt: Mapping[str, Any]) -> tuple[dict[str, str], bytes, list[str]]:
    if set(receipt) != {
        "protocol", "authority_sha256", "bindings", "manifest_base64", "launcher_command", "receipt_sha256"
    } or receipt.get("protocol") != PROTOCOL:
        raise DeploymentRefused("invalid receipt schema or protocol")
    body = {key: receipt[key] for key in (
        "protocol", "authority_sha256", "bindings", "manifest_base64", "launcher_command"
    )}
    if receipt["receipt_sha256"] != _bytes_sha256(_canonical(body)):
        raise DeploymentRefused("receipt SHA mismatch")
    bindings = receipt["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != set(FIELDS):
        raise DeploymentRefused("receipt must bind exactly six identities")
    if any(not isinstance(bindings[field], str) or not bindings[field] for field in FIELDS):
        raise DeploymentRefused("receipt identity values must be non-empty strings")
    try:
        manifest_bytes = base64.b64decode(receipt["manifest_base64"], validate=True)
        json.loads(manifest_bytes)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DeploymentRefused("invalid embedded manifest") from exc
    command = receipt["launcher_command"]
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part for part in command
    ):
        raise DeploymentRefused("invalid launcher command")
    return bindings, manifest_bytes, command


def launch_receipt(
    receipt: Mapping[str, Any],
    *,
    inputs: LiveInputs,
    ledger_path: Path,
    authorized_receipt_sha256: str,
    receipt_bytes: bytes | None = None,
    launcher: Callable[[Sequence[str]], int],
    probes: Probes = DEFAULT_PROBES,
) -> int:
    """Invoke ``launcher`` exactly once only after all six live checks match."""
    _append_event(ledger_path, {"event": "deployment_intent", "protocol": PROTOCOL})
    try:
        artifact = receipt_bytes if receipt_bytes is not None else receipt_artifact_bytes(receipt)
        if receipt_bytes is not None and receipt_bytes != receipt_artifact_bytes(receipt):
            raise DeploymentRefused("receipt artifact is not canonical")
        if _bytes_sha256(artifact) != authorized_receipt_sha256:
            raise DeploymentRefused("external authorized receipt SHA mismatch")
        expected, manifest_bytes, command = _unpack_receipt(receipt)
        observed = _observed(inputs, manifest_bytes, probes)
        mismatches = [field for field in FIELDS if observed[field] != expected[field]]
    except (DeploymentRefused, OSError, subprocess.SubprocessError, ValueError) as exc:
        _append_event(ledger_path, {"decision": "REJECT", "event": "prelaunch_decision", "reason": str(exc)})
        return 2
    if mismatches:
        _append_event(ledger_path, {"decision": "REJECT", "event": "prelaunch_decision", "mismatches": mismatches})
        return 2
    _append_event(ledger_path, {"decision": "ACCEPT", "event": "prelaunch_decision", "mismatches": []})
    _append_event(ledger_path, {"event": "launch_attempt"})
    return int(launcher(tuple(command)))


def _run(command: Sequence[str]) -> int:
    return subprocess.run(list(command), check=False).returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create-receipt")
    create.add_argument("--control-root", type=Path, required=True)
    create.add_argument("--runtime-root", type=Path, required=True)
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--authority", type=Path, required=True)
    create.add_argument("--authority-sha256", required=True)
    create.add_argument("--model-path", type=Path, required=True)
    create.add_argument("--extension", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("launcher", nargs=argparse.REMAINDER)
    launch = sub.add_parser("launch")
    launch.add_argument("--receipt", type=Path, required=True)
    launch.add_argument("--authorized-receipt-sha256", required=True)
    launch.add_argument("--control-root", type=Path, required=True)
    launch.add_argument("--runtime-root", type=Path, required=True)
    launch.add_argument("--model-path", type=Path, required=True)
    launch.add_argument("--extension", type=Path, required=True)
    launch.add_argument("--ledger", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = LiveInputs(args.control_root, args.runtime_root, args.model_path, args.extension)
    if args.action == "create-receipt":
        command = args.launcher[1:] if args.launcher[:1] == ["--"] else args.launcher
        authority = json.loads(args.authority.read_text(encoding="utf-8"))
        receipt = build_receipt(
            inputs=inputs,
            manifest_path=args.manifest,
            launcher_command=command,
            authority=authority,
            authorized_authority_sha256=args.authority_sha256,
        )
        args.output.write_bytes(receipt_artifact_bytes(receipt))
        return 0
    receipt_bytes = args.receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    return launch_receipt(
        receipt,
        inputs=inputs,
        ledger_path=args.ledger,
        authorized_receipt_sha256=args.authorized_receipt_sha256,
        receipt_bytes=receipt_bytes,
        launcher=_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Sole interpreter-bound, fail-closed deployment gate for RIST C0 v3."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROTOCOL = "RIST-C0-v3.0-DEPLOYMENT-RECEIPT-v1"
AUTHORITY_PROTOCOL = "RIST-C0-v3.0-DEPLOYMENT-AUTHORITY-v1"
FIELDS = (
    "control_commit",
    "runtime_commit",
    "manifest_sha256",
    "model_revision",
    "gpu_uuid",
    "extension_sha256",
    "interpreter_realpath",
    "interpreter_sha256",
    "interpreter_version",
)


class DeploymentRefused(RuntimeError):
    """Raised before any request-capable process is started."""


@dataclass(frozen=True)
class LiveInputs:
    source_root: Path
    model_path: Path
    extension_path: Path
    interpreter_path: Path


@dataclass(frozen=True)
class Probes:
    git_head: Callable[[Path], str]
    model_revision: Callable[[Path], str]
    gpu_uuid: Callable[[], str]
    file_sha256: Callable[[Path], str]
    interpreter_version: Callable[[Path], str]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _interpreter_version(path: Path) -> str:
    return subprocess.check_output(
        [str(path.resolve(strict=True)), "--version"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


DEFAULT_PROBES = Probes(
    _git_head, _model_revision, _gpu_uuid, _file_sha256, _interpreter_version
)


def receipt_artifact_bytes(receipt: Mapping[str, Any]) -> bytes:
    return _canonical(receipt) + b"\n"


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, _canonical(dict(event)) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _observed(
    inputs: LiveInputs, manifest_bytes: bytes, probes: Probes
) -> dict[str, str]:
    source_root = inputs.source_root.resolve(strict=True)
    interpreter = inputs.interpreter_path.resolve(strict=True)
    commit = probes.git_head(source_root)
    return {
        "control_commit": commit,
        "runtime_commit": commit,
        "manifest_sha256": _bytes_sha256(manifest_bytes),
        "model_revision": probes.model_revision(inputs.model_path),
        "gpu_uuid": probes.gpu_uuid(),
        "extension_sha256": probes.file_sha256(inputs.extension_path),
        "interpreter_realpath": str(interpreter),
        "interpreter_sha256": probes.file_sha256(interpreter),
        "interpreter_version": probes.interpreter_version(interpreter),
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
    source_root = inputs.source_root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    try:
        manifest_path.relative_to(source_root)
    except ValueError as exc:
        raise DeploymentRefused("manifest must be inside the sole source worktree") from exc
    manifest_bytes = manifest_path.read_bytes()
    json.loads(manifest_bytes)
    observed = _observed(inputs, manifest_bytes, probes)
    if not launcher_command or launcher_command[0] != observed["interpreter_realpath"]:
        raise DeploymentRefused("launcher must begin with the bound interpreter real path")
    if not all(isinstance(part, str) and part for part in launcher_command):
        raise DeploymentRefused("launcher command must contain non-empty strings")
    authority_sha = _bytes_sha256(_canonical(authority))
    if authority_sha != authorized_authority_sha256:
        raise DeploymentRefused("deployment authority SHA mismatch")
    expected_keys = {"protocol", *FIELDS}
    if set(authority) != expected_keys or authority.get("protocol") != AUTHORITY_PROTOCOL:
        raise DeploymentRefused("invalid deployment authority schema or protocol")
    mismatches = [field for field in FIELDS if authority[field] != observed[field]]
    if mismatches:
        raise DeploymentRefused("authority mismatch: " + ", ".join(mismatches))
    body = {
        "protocol": PROTOCOL,
        "authority_sha256": authority_sha,
        "bindings": observed,
        "manifest_base64": base64.b64encode(manifest_bytes).decode("ascii"),
        "launcher_command": list(launcher_command),
    }
    return {**body, "receipt_sha256": _bytes_sha256(_canonical(body))}


def _unpack(receipt: Mapping[str, Any]) -> tuple[dict[str, str], bytes, list[str]]:
    expected = {
        "protocol", "authority_sha256", "bindings", "manifest_base64",
        "launcher_command", "receipt_sha256",
    }
    if set(receipt) != expected or receipt.get("protocol") != PROTOCOL:
        raise DeploymentRefused("invalid receipt schema or protocol")
    body = {key: receipt[key] for key in expected if key != "receipt_sha256"}
    if receipt["receipt_sha256"] != _bytes_sha256(_canonical(body)):
        raise DeploymentRefused("receipt self-hash mismatch")
    bindings = receipt["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != set(FIELDS):
        raise DeploymentRefused("receipt must bind exactly the v3 identities")
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
    _append_event(ledger_path, {"event": "deployment_intent", "protocol": PROTOCOL})
    try:
        artifact = receipt_artifact_bytes(receipt)
        if receipt_bytes is not None and receipt_bytes != artifact:
            raise DeploymentRefused("receipt artifact is not canonical")
        if _bytes_sha256(artifact) != authorized_receipt_sha256:
            raise DeploymentRefused("external authorized receipt SHA mismatch")
        expected, manifest_bytes, command = _unpack(receipt)
        observed = _observed(inputs, manifest_bytes, probes)
        mismatches = [field for field in FIELDS if observed[field] != expected[field]]
        if command[0] != observed["interpreter_realpath"]:
            mismatches.append("launcher_interpreter")
    except (DeploymentRefused, OSError, subprocess.SubprocessError, ValueError) as exc:
        _append_event(
            ledger_path,
            {"event": "prelaunch_decision", "decision": "REJECT", "reason": str(exc)},
        )
        return 2
    if mismatches:
        _append_event(
            ledger_path,
            {"event": "prelaunch_decision", "decision": "REJECT", "mismatches": mismatches},
        )
        return 2
    _append_event(
        ledger_path,
        {"event": "prelaunch_decision", "decision": "ACCEPT", "mismatches": []},
    )
    _append_event(ledger_path, {"event": "launch_attempt"})
    return int(launcher(tuple(command)))


def _run_supervised(command: Sequence[str]) -> int:
    child = subprocess.Popen(list(command))
    prior = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    def forward(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    try:
        for signum in prior:
            signal.signal(signum, forward)
        return int(child.wait())
    finally:
        for signum, handler in prior.items():
            signal.signal(signum, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--authorized-receipt-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    receipt_bytes = args.receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    return launch_receipt(
        receipt,
        inputs=LiveInputs(
            args.source_root, args.model_path, args.extension, Path(sys.executable)
        ),
        ledger_path=args.ledger,
        authorized_receipt_sha256=args.authorized_receipt_sha256,
        receipt_bytes=receipt_bytes,
        launcher=_run_supervised,
    )


if __name__ == "__main__":
    raise SystemExit(main())

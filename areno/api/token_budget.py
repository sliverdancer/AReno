"""Strict completed-optimizer-step accounting for token-matched RL runs."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "areno.max_trainable_tokens.v1"


def rng_state_identity() -> dict[str, Any]:
    """Hash parent-process RNG states without serialising restorable state."""

    states: dict[str, str] = {"python": repr(random.getstate())}
    try:
        import numpy as np

        states["numpy"] = repr(np.random.get_state())
    except ImportError:
        pass
    try:
        import torch

        states["torch_cpu"] = bytes(torch.random.get_rng_state().tolist()).hex()
    except ImportError:
        pass
    payload = json.dumps(states, sort_keys=True, separators=(",", ":")).encode()
    return {
        "scope": "parent_process_only",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "components": sorted(states),
        "worker_rng_state_saved": False,
    }


def config_identity(config: Any) -> str:
    """Return a stable digest of the public trainer configuration."""

    values = asdict(config) if is_dataclass(config) else vars(config)
    payload = json.dumps(values, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def checkpoint_identity(path: str) -> dict[str, Any]:
    """Hash relative paths, sizes, and contents of a completed checkpoint."""

    root = Path(path)
    if not root.is_dir():
        raise RuntimeError(f"checkpoint path is not a directory: {path}")
    manifest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = file_path.relative_to(root).as_posix()
        content_hash = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                content_hash.update(chunk)
                total_bytes += len(chunk)
        manifest.update(f"{relative}\0{file_path.stat().st_size}\0{content_hash.hexdigest()}\n".encode())
        files += 1
    if files == 0:
        raise RuntimeError(f"checkpoint contains no files: {path}")
    return {
        "path": str(root.resolve()),
        "manifest_sha256": manifest.hexdigest(),
        "file_count": files,
        "total_bytes": total_bytes,
    }


def records_identity(records: list[dict[str, Any]]) -> str:
    """Bind the accepted terminal batch records without exposing their text."""

    payload = json.dumps(records, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_terminal_evidence(path: str, *, verify_checkpoint: bool = True) -> dict[str, Any]:
    """Independently validate a reached token-budget terminal sidecar."""

    evidence = json.loads(Path(path).read_text(encoding="utf-8"))
    if evidence.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("token-budget protocol version mismatch")
    if evidence.get("status") != "TARGET_REACHED":
        raise ValueError("terminal evidence is not TARGET_REACHED")
    integer_fields = [
        "target",
        "tokens_before_step",
        "terminal_step_tokens",
        "tokens_after_step",
        "overshoot",
        "optimizer_global_step",
        "optimizer_steps_completed",
    ]
    values = {field: _exact_non_negative_int(evidence.get(field), field) for field in integer_fields}
    if values["optimizer_steps_completed"] != 1:
        raise ValueError("terminal evidence must bind exactly one optimizer step")
    if values["terminal_step_tokens"] <= 0:
        raise ValueError("terminal optimizer step must contain trainable tokens")
    if values["tokens_before_step"] + values["terminal_step_tokens"] != values["tokens_after_step"]:
        raise ValueError("terminal token accounting equation failed")
    if values["tokens_before_step"] >= values["target"] or values["tokens_after_step"] < values["target"]:
        raise ValueError("terminal step is not the first target-crossing step")
    if values["overshoot"] != values["tokens_after_step"] - values["target"]:
        raise ValueError("terminal overshoot equation failed")
    if verify_checkpoint:
        recorded = evidence.get("checkpoint")
        if not isinstance(recorded, dict):
            raise ValueError("terminal checkpoint identity is missing")
        actual = checkpoint_identity(str(recorded.get("path")))
        if actual != recorded:
            raise ValueError("terminal checkpoint identity mismatch")
    return evidence


class TokenBudgetTracker:
    """Accept only verified, exactly-one-optimizer-step accounting events."""

    def __init__(self, target: int):
        self.target = target
        self.completed = 0
        self.skipped_batches = 0
        self._last_global_step: int | None = None

    @property
    def last_global_step(self) -> int | None:
        return self._last_global_step

    def consume(self, result: dict[str, Any]) -> dict[str, int | bool]:
        required = {"completed_trainable_tokens", "optimizer_steps_completed"}
        missing = sorted(required.difference(result))
        if missing:
            raise RuntimeError(f"token-budget backend result missing: {', '.join(missing)}")
        step_tokens = _exact_non_negative_int(result["completed_trainable_tokens"], "completed_trainable_tokens")
        optimizer_steps = _exact_non_negative_int(result["optimizer_steps_completed"], "optimizer_steps_completed")
        if optimizer_steps == 0 and step_tokens == 0:
            self.skipped_batches += 1
            return {
                "target": self.target,
                "tokens_before_step": self.completed,
                "terminal_step_tokens": 0,
                "tokens_after_step": self.completed,
                "overshoot": max(self.completed - self.target, 0),
                "optimizer_global_step": self._last_global_step,
                "target_reached": False,
                "optimizer_step_skipped": True,
                "optimizer_step_skipped_batches": self.skipped_batches,
            }
        if optimizer_steps != 1:
            raise RuntimeError("token-budget mode requires exactly one completed optimizer step per train call")
        if "optimizer_global_step" not in result:
            raise RuntimeError("token-budget backend result missing: optimizer_global_step")
        global_step = _exact_non_negative_int(result["optimizer_global_step"], "optimizer_global_step")
        if step_tokens <= 0:
            raise RuntimeError("completed optimizer step has no post-mask trainable tokens")
        if self._last_global_step is None and global_step != 1:
            raise RuntimeError("fresh token-budget run must begin at optimizer global step 1")
        if self._last_global_step is not None and global_step != self._last_global_step + 1:
            raise RuntimeError("optimizer global-step identity is not contiguous")
        before = self.completed
        self.completed += step_tokens
        self._last_global_step = global_step
        return {
            "target": self.target,
            "tokens_before_step": before,
            "terminal_step_tokens": step_tokens,
            "tokens_after_step": self.completed,
            "overshoot": max(self.completed - self.target, 0),
            "optimizer_global_step": global_step,
            "target_reached": self.completed >= self.target,
            "optimizer_step_skipped": False,
            "optimizer_step_skipped_batches": self.skipped_batches,
        }


def assert_token_budget_outputs_available(
    metrics_directory: str,
    checkpoint_directory: str,
    target: int,
) -> None:
    """Fail before runtime initialization when prior terminal outputs exist."""

    destination = Path(metrics_directory) / "token_budget_terminal.json"
    if destination.exists():
        raise RuntimeError(f"token-budget terminal evidence already exists: {destination}")
    checkpoint_root = Path(checkpoint_directory)
    pattern = f"token_budget_target_{target}_optimizer_step_*"
    conflicts = sorted(checkpoint_root.glob(pattern)) if checkpoint_root.exists() else []
    if conflicts:
        raise RuntimeError(f"token-budget terminal checkpoint already exists: {conflicts[0]}")


def write_terminal_evidence(directory: str, evidence: dict[str, Any]) -> Path:
    """Create terminal evidence atomically and refuse ambiguous overwrites."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "token_budget_terminal.json"
    if destination.exists():
        raise RuntimeError(f"token-budget terminal evidence already exists: {destination}")
    temporary = root / f".{destination.name}.{os.getpid()}.tmp"
    payload = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def _exact_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be an exact non-negative integer")
    integer = int(value)
    if integer < 0 or float(value) != integer:
        raise RuntimeError(f"{name} must be an exact non-negative integer")
    return integer

from __future__ import annotations

import importlib.util
import hashlib
import signal
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / (
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/"
    "stages/C0_RESOLUTION_V2_3"
)


def _load():
    stage = str(STAGE)
    sys.path.insert(0, stage)
    try:
        path = STAGE / "graceful_deployment_entrypoint.py"
        spec = importlib.util.spec_from_file_location("rist_c0_v23_graceful", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(stage)


def _load_verifier():
    path = STAGE / "verify_clean_shutdown_cpu_freeze.py"
    spec = importlib.util.spec_from_file_location("rist_c0_v23_clean_verify", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeChild:
    def __init__(self, command, handlers, *, returncode=0, wait_error=None):
        self.command = command
        self.handlers = handlers
        self.returncode = returncode
        self.wait_error = wait_error
        self.signals = []
        self.running = True

    def poll(self):
        return None if self.running else self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)
        self.running = False

    def wait(self):
        self.handlers[signal.SIGTERM](signal.SIGTERM, None)
        if self.wait_error is not None:
            raise self.wait_error
        return self.returncode


def _install_fakes(module, monkeypatch, *, returncode=0, wait_error=None):
    handlers = {}
    restored = []
    original = {signal.SIGINT: object(), signal.SIGTERM: object()}
    children = []

    def fake_signal(signum, handler):
        if handler is original[signum]:
            restored.append(signum)
        else:
            handlers[signum] = handler

    def fake_popen(command):
        child = _FakeChild(
            command,
            handlers,
            returncode=returncode,
            wait_error=wait_error,
        )
        children.append(child)
        return child

    monkeypatch.setattr(module.signal, "getsignal", lambda signum: original[signum])
    monkeypatch.setattr(module.signal, "signal", fake_signal)
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    return children, restored


def test_graceful_supervisor_forwards_term_only_to_exact_server_child(monkeypatch):
    module = _load()
    children, restored = _install_fakes(module, monkeypatch, returncode=17)

    result = module._run_supervised(("python", "-m", "areno.cli.main", "serve"))

    assert result == 17
    assert len(children) == 1
    assert children[0].command == ["python", "-m", "areno.cli.main", "serve"]
    assert children[0].signals == [signal.SIGTERM]
    assert restored == [signal.SIGINT, signal.SIGTERM]
    assert not hasattr(module, "os")


def test_graceful_supervisor_restores_handlers_when_child_wait_fails(monkeypatch):
    module = _load()
    children, restored = _install_fakes(
        module,
        monkeypatch,
        wait_error=RuntimeError("synthetic wait failure"),
    )

    with pytest.raises(RuntimeError, match="synthetic wait failure"):
        module._run_supervised(("fake", "serve"))

    assert children[0].signals == [signal.SIGTERM]
    assert restored == [signal.SIGINT, signal.SIGTERM]


def test_clean_shutdown_successor_preserves_consumed_capacity_artifacts():
    verifier = _load_verifier()
    original_entrypoint = STAGE / "deployment_entrypoint.py"
    capacity_result = (
        STAGE
        / "gpu_bind_20260804_a800/capacity_20260804_v1/CAPACITY_CANARY_RESULT.json"
    )

    assert hashlib.sha256(original_entrypoint.read_bytes()).hexdigest() == (
        verifier.ORIGINAL_ENTRYPOINT_SHA256
    )
    assert hashlib.sha256(capacity_result.read_bytes()).hexdigest() == (
        verifier.CAPACITY_RESULT_SHA256
    )

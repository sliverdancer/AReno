from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / (
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/"
    "stages/C0_RESOLUTION_V2_3/deployment_entrypoint.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("rist_c0_v23_deploy", ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _setup(tmp_path: Path):
    module = _load()
    control = tmp_path / "control"
    runtime = tmp_path / "runtime"
    model = tmp_path / "models" / "snapshots" / ("c" * 40)
    for path in (control, runtime, model):
        path.mkdir(parents=True)
    manifest = control / "manifest.json"
    manifest.write_text('{"frozen":true}\n', encoding="utf-8")
    extension = tmp_path / "extension.so"
    extension.write_bytes(b"extension")
    values = {control.resolve(): "a" * 40, runtime.resolve(): "b" * 40}
    live = {"gpu": "GPU-test"}
    probes = module.Probes(
        git_head=lambda path: values[path.resolve()],
        model_revision=lambda path: path.resolve().name,
        gpu_uuid=lambda: live["gpu"],
        file_sha256=module._file_sha256,
    )
    inputs = module.LiveInputs(control, runtime, model, extension)
    receipt = module.build_receipt(
        inputs=inputs,
        manifest_path=manifest,
        launcher_command=("fake", "serve"),
        probes=probes,
    )
    return module, inputs, probes, receipt, values, live, manifest, extension


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run(module, inputs, probes, receipt, ledger):
    calls = []
    result = module.launch_receipt(
        receipt,
        inputs=inputs,
        ledger_path=ledger,
        probes=probes,
        launcher=lambda command: calls.append(tuple(command)) or 0,
    )
    return result, calls, _events(ledger)


@pytest.mark.parametrize(
    "field",
    ("control_commit", "runtime_commit", "manifest_sha256", "model_revision", "gpu_uuid", "extension_sha256"),
)
def test_dev_replay_each_identity_mismatch_has_zero_launch(tmp_path, field):
    module, inputs, probes, receipt, values, live, _, extension = _setup(tmp_path)
    if field == "control_commit":
        values[inputs.control_root.resolve()] = "d" * 40
    elif field == "runtime_commit":
        values[inputs.runtime_root.resolve()] = "d" * 40
    elif field == "manifest_sha256":
        receipt["bindings"][field] = "d" * 64
        body = {key: receipt[key] for key in ("protocol", "bindings", "manifest_base64", "launcher_command")}
        receipt["receipt_sha256"] = module._bytes_sha256(module._canonical(body))
    elif field == "model_revision":
        inputs = module.LiveInputs(inputs.control_root, inputs.runtime_root, tmp_path / "models/snapshots/wrong", extension)
        inputs.model_path.mkdir(parents=True)
    elif field == "gpu_uuid":
        live["gpu"] = "GPU-wrong"
    else:
        extension.write_bytes(b"changed")
    result, calls, events = _run(module, inputs, probes, receipt, tmp_path / "ledger.jsonl")
    assert result == 2
    assert calls == []
    assert events[-1]["mismatches"] == [field]
    assert not any(event["event"] == "launch_attempt" for event in events)


def test_dev_replay_wrong_runtime_worktree_and_combined_mismatch_cannot_launch(tmp_path):
    module, inputs, probes, receipt, values, live, _, extension = _setup(tmp_path)
    wrong_runtime = tmp_path / "wrong-runtime"
    wrong_runtime.mkdir()
    values[wrong_runtime.resolve()] = "e" * 40
    live["gpu"] = "GPU-wrong"
    extension.write_bytes(b"wrong-extension")
    wrong = module.LiveInputs(inputs.control_root, wrong_runtime, inputs.model_path, extension)
    result, calls, events = _run(module, wrong, probes, receipt, tmp_path / "ledger.jsonl")
    assert result == 2
    assert calls == []
    assert set(events[-1]["mismatches"]) == {"runtime_commit", "gpu_uuid", "extension_sha256"}
    assert [event["event"] for event in events] == ["deployment_intent", "prelaunch_decision"]


def test_dev_replay_receipt_manifest_tamper_cannot_launch(tmp_path):
    module, inputs, probes, receipt, _, _, _, _ = _setup(tmp_path)
    receipt["manifest_base64"] = "e30="
    result, calls, events = _run(module, inputs, probes, receipt, tmp_path / "ledger.jsonl")
    assert result == 2
    assert calls == []
    assert events[-1]["reason"] == "receipt SHA mismatch"
    assert not any(event["event"] == "launch_attempt" for event in events)


def test_dev_replay_success_launches_exactly_once_and_appends_ledger(tmp_path):
    module, inputs, probes, receipt, _, _, _, _ = _setup(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"event":"prior"}\n', encoding="utf-8")
    result, calls, events = _run(module, inputs, probes, receipt, ledger)
    assert result == 0
    assert calls == [("fake", "serve")]
    assert [event["event"] for event in events] == [
        "prior", "deployment_intent", "prelaunch_decision", "launch_attempt"
    ]


def test_merge_gate_wrong_worktrees_cannot_self_authorize_a_receipt(tmp_path):
    module, inputs, probes, _receipt, values, _live, manifest, _extension = _setup(tmp_path)
    wrong_control = tmp_path / "wrong-control"
    wrong_runtime = tmp_path / "wrong-runtime"
    wrong_control.mkdir()
    wrong_runtime.mkdir()
    wrong_manifest = wrong_control / "manifest.json"
    wrong_manifest.write_bytes(manifest.read_bytes())
    values[wrong_control.resolve()] = "d" * 40
    values[wrong_runtime.resolve()] = "e" * 40
    wrong_inputs = module.LiveInputs(
        wrong_control, wrong_runtime, inputs.model_path, inputs.extension_path
    )
    receipt = module.build_receipt(
        inputs=wrong_inputs,
        manifest_path=wrong_manifest,
        launcher_command=("fake", "serve"),
        probes=probes,
    )
    calls = []

    result = module.launch_receipt(
        receipt,
        inputs=wrong_inputs,
        ledger_path=tmp_path / "ledger.jsonl",
        probes=probes,
        launcher=lambda command: calls.append(tuple(command)) or 0,
    )

    assert result == 2
    assert calls == []


def test_merge_gate_recomputed_self_hash_is_not_deployment_authority(tmp_path):
    module, inputs, probes, receipt, values, _live, _manifest, _extension = _setup(tmp_path)
    values[inputs.runtime_root.resolve()] = "e" * 40
    receipt["bindings"]["runtime_commit"] = "e" * 40
    body = {
        key: receipt[key]
        for key in ("protocol", "bindings", "manifest_base64", "launcher_command")
    }
    receipt["receipt_sha256"] = module._bytes_sha256(module._canonical(body))
    calls = []

    result = module.launch_receipt(
        receipt,
        inputs=inputs,
        ledger_path=tmp_path / "ledger.jsonl",
        probes=probes,
        launcher=lambda command: calls.append(tuple(command)) or 0,
    )

    assert result == 2
    assert calls == []

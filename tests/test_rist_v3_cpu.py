from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "research/reward_identifiability_supervision_topology/successors/rist_v3"
STAGE = V3 / "stages/C0_RESOLUTION"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _replay(tmp_path: Path, mode: str) -> dict:
    output = tmp_path / f"{mode}.json"
    subprocess.run(
        [
            sys.executable,
            str(STAGE / "replay_deployment.py"),
            "--mode", mode,
            "--output", str(output),
        ],
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_v3_pool_is_fresh_balanced_and_seed_disjoint():
    builder = _load("rist_v3_pool", STAGE / "build_fresh_pool.py")
    rows = builder.build_rows()
    assert {key: len(value) for key, value in rows.items()} == {
        "calibration": 32, "qualification": 32, "capacity_canary": 1,
    }
    signatures = [row["task_signature"] for values in rows.values() for row in values]
    prior_signatures, prior_seeds = builder._prior_values()
    assert not set(signatures) & prior_signatures
    v3_seeds = {builder.POOL_SEED, builder.CANARY_TASK_SEED}.union(
        *(set(values) for values in builder.ROLLOUT_SEEDS.values())
    )
    assert not v3_seeds & prior_seeds


def test_v3_receipt_binds_interpreter_and_has_no_path_lookup(tmp_path):
    gate = _load("rist_v3_gate", STAGE / "deployment_entrypoint.py")
    source = tmp_path / "source"
    model = tmp_path / "models/snapshots" / ("c" * 40)
    source.mkdir()
    model.mkdir(parents=True)
    manifest = source / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    extension = tmp_path / "extension.so"
    extension.write_bytes(b"extension")
    interpreter = Path(sys.executable).resolve()
    probes = gate.Probes(
        git_head=lambda _path: "a" * 40,
        model_revision=lambda path: path.resolve().name,
        gpu_uuid=lambda: "GPU-test",
        file_sha256=gate._file_sha256,
        interpreter_version=gate._interpreter_version,
    )
    inputs = gate.LiveInputs(source, model, extension, interpreter)
    observed = gate._observed(inputs, manifest.read_bytes(), probes)
    authority = {"protocol": gate.AUTHORITY_PROTOCOL, **observed}
    receipt = gate.build_receipt(
        inputs=inputs,
        manifest_path=manifest,
        launcher_command=(str(interpreter), "-c", "pass"),
        authority=authority,
        authorized_authority_sha256=gate._bytes_sha256(gate._canonical(authority)),
        probes=probes,
    )
    assert receipt["bindings"]["interpreter_realpath"] == str(interpreter)
    assert len(receipt["bindings"]["interpreter_sha256"]) == 64
    assert receipt["launcher_command"][0] == str(interpreter)
    with pytest.raises(gate.DeploymentRefused, match="bound interpreter"):
        gate.build_receipt(
            inputs=inputs,
            manifest_path=manifest,
            launcher_command=("unbound-interpreter", "-c", "pass"),
            authority=authority,
            authorized_authority_sha256=gate._bytes_sha256(gate._canonical(authority)),
            probes=probes,
        )


@pytest.mark.parametrize("mode", ("wrong_worktree", "wrong_interpreter"))
def test_v3_exact_replay_rejects_before_child_launch(tmp_path, mode):
    result = _replay(tmp_path, mode)
    assert result["decision"] == "REJECT"
    assert result["launch_attempt_count"] == 0
    assert result["marker_created"] is False


def test_v3_exact_replay_launches_with_bound_interpreter(tmp_path):
    result = _replay(tmp_path, "success")
    assert result["decision"] == "ACCEPT"
    assert result["launch_attempt_count"] == 1
    assert result["marker_created"] is True
    assert result["model_accessed"] is False
    assert result["gpu_used"] is False


def test_v3_code_contains_no_hardcoded_interpreter_lookup():
    for path in STAGE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert 'Popen(["python' not in text
        assert 'run(["python' not in text
        assert '"python3"' not in text


def test_v3_calibration_manifest_does_not_read_qualification(tmp_path, monkeypatch):
    builder = _load("rist_v3_stage_cal", STAGE / "build_stage_manifest.py")
    pool = STAGE / "data/manifest.json"
    qualification = (STAGE / "data/qualification.jsonl").resolve()
    seen = []
    original = builder._sha256

    def tracking(path):
        seen.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(builder, "_sha256", tracking)
    manifest = builder.build_manifest(
        split="calibration",
        pool_manifest_path=pool,
        output_root=tmp_path / "results",
        source_commit="a" * 40,
    )
    assert manifest["trajectory_count"] == 2048
    assert manifest["qualification_permitted"] is False
    assert qualification not in seen


def test_v3_qualification_requires_exact_admission(tmp_path):
    builder = _load("rist_v3_stage_qual", STAGE / "build_stage_manifest.py")
    pool = STAGE / "data/manifest.json"
    kwargs = dict(
        split="qualification",
        pool_manifest_path=pool,
        output_root=tmp_path / "results",
        source_commit="a" * 40,
    )
    with pytest.raises(PermissionError, match="sealed"):
        builder.build_manifest(**kwargs)
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({
        "protocol": builder.ADMISSION_PROTOCOL,
        "decision": "PASS_CALIBRATION_TO_QUALIFICATION",
        "passed": True,
        "pool_manifest_sha256": builder._sha256(pool),
        "calibration_result_sha256": "b" * 64,
        "frozen_whole_cell_map": {
            "c00": "low", "c01": "low", "c04": "high", "c05": "high",
        },
        "qualification_accessed": False,
    }, sort_keys=True) + "\n", encoding="utf-8")
    manifest = builder.build_manifest(
        **kwargs, calibration_admission_path=admission
    )
    assert manifest["qualification_permitted"] is True
    assert manifest["calibration_permitted"] is False
    assert manifest["training_permitted"] is False

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "research/reward_identifiability_supervision_topology/successors/rist_v3"
V4 = ROOT / "research/reward_identifiability_supervision_topology/successors/rist_v4"
STAGE = V4 / "stages/C0_RESOLUTION"


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
        [sys.executable, str(STAGE / "replay_deployment.py"), "--mode", mode, "--output", str(output)],
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_v4_pool_is_balanced_seed_disjoint_and_semantic():
    builder = _load("rist_v4_pool", STAGE / "build_fresh_pool.py")
    rows = builder.build_rows()
    assert {key: len(value) for key, value in rows.items()} == {
        "calibration": 32,
        "qualification": 32,
        "capacity_canary": 1,
    }
    signatures = [row["task_signature"] for values in rows.values() for row in values]
    prior_signatures, prior_seeds = builder._prior_values()
    assert len(signatures) == len(set(signatures))
    assert not set(signatures) & prior_signatures
    v4_seeds = {builder.POOL_SEED, builder.CANARY_TASK_SEED}.union(
        *(set(values) for values in builder.ROLLOUT_SEEDS.values())
    )
    assert not v4_seeds & prior_seeds
    for row in rows["calibration"] + rows["qualification"]:
        prompt = json.dumps(row["prompt_contract"], sort_keys=True)
        assert "reward_resolution" not in prompt
        assert "strict_success" not in prompt
        assert "anchor_intent" not in prompt
        assert row["generator_version"] == builder.GENERATOR_VERSION
        for turn in row["turns"]:
            assert turn["offered_tools"].count(turn["expected_tool"]) == 1
            for candidate in turn["candidate_records"]:
                assert 2 <= len(candidate["code"]) <= 3
                assert candidate["code"][0].isalpha()
                assert candidate["code"][1:].isdigit()


def test_v4_written_manifest_hashes_match():
    manifest_path = STAGE / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["protocol"] == "RIST-C0-v4.0-FRESH-POOL-v1"
    assert manifest["v3_outcomes_used_for_task_selection"] is False
    assert manifest["model_accessed"] is False
    assert manifest["gpu_used"] is False
    for split, spec in manifest["splits"].items():
        path = manifest_path.parent / spec["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]
        assert spec["structural_stats"]["max_code_length"] <= 3
        assert spec["task_count"] == (1 if split == "capacity_canary" else 32)


def test_v4_calibration_manifest_does_not_read_qualification(tmp_path, monkeypatch):
    builder = _load("rist_v4_stage_cal", STAGE / "build_stage_manifest.py")
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
    assert manifest["protocol"] == "RIST-C0-v4.0-STAGE-MANIFEST-v1"
    assert manifest["trajectory_count"] == 2048
    assert manifest["qualification_permitted"] is False
    assert qualification not in seen


def test_v4_capacity_canary_manifest_is_small_and_sealed(tmp_path, monkeypatch):
    builder = _load("rist_v4_stage_canary", STAGE / "build_stage_manifest.py")
    pool = STAGE / "data/manifest.json"
    protected = {
        (STAGE / "data/calibration.jsonl").resolve(),
        (STAGE / "data/qualification.jsonl").resolve(),
    }
    seen = []
    original = builder._sha256

    def tracking(path):
        seen.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(builder, "_sha256", tracking)
    manifest = builder.build_manifest(
        split="capacity_canary",
        pool_manifest_path=pool,
        output_root=tmp_path / "results",
        source_commit="a" * 40,
    )
    assert manifest["protocol"] == "RIST-C0-v4.0-STAGE-MANIFEST-v1"
    assert manifest["trajectory_count"] == 16
    assert manifest["capacity_canary_permitted"] is True
    assert manifest["calibration_permitted"] is False
    assert manifest["qualification_permitted"] is False
    assert manifest["training_permitted"] is False
    assert manifest["retry_permitted"] is False
    assert not (protected & set(seen))
    assert {job["trajectory_count"] for job in manifest["jobs"]} == {8}
    assert {job["job_id"] for job in manifest["jobs"]} == {
        "qwen3-capacity_canary",
        "gemma4-capacity_canary",
    }


def test_v4_qualification_requires_exact_admission(tmp_path):
    builder = _load("rist_v4_stage_qual", STAGE / "build_stage_manifest.py")
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
        "frozen_whole_cell_map": {"c00": "low", "c01": "low", "c06": "high", "c07": "high"},
        "qualification_accessed": False,
    }, sort_keys=True) + "\n", encoding="utf-8")
    manifest = builder.build_manifest(**kwargs, calibration_admission_path=admission)
    assert manifest["qualification_permitted"] is True
    assert manifest["calibration_permitted"] is False
    assert manifest["training_permitted"] is False


@pytest.mark.parametrize("mode", ("wrong_worktree", "wrong_interpreter"))
def test_v4_exact_replay_rejects_before_child_launch(tmp_path, mode):
    result = _replay(tmp_path, mode)
    assert result["decision"] == "REJECT"
    assert result["launch_attempt_count"] == 0
    assert result["marker_created"] is False
    assert result["model_accessed"] is False
    assert result["gpu_used"] is False


def test_v4_exact_replay_launches_with_bound_interpreter(tmp_path):
    result = _replay(tmp_path, "success")
    assert result["decision"] == "ACCEPT"
    assert result["launch_attempt_count"] == 1
    assert result["marker_created"] is True
    assert result["model_accessed"] is False
    assert result["gpu_used"] is False


def test_v4_binder_is_non_circular(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(STAGE))
    binder = _load("rist_v4_binder", STAGE / "bind_manifest_runtime_identities.py")
    gate = binder.gate
    builder = _load("rist_v4_stage_bind", STAGE / "build_stage_manifest.py")
    manifest = builder.build_manifest(
        split="calibration",
        pool_manifest_path=STAGE / "data/manifest.json",
        output_root=tmp_path / "results",
        source_commit="a" * 40,
    )
    source = tmp_path / "source"
    qwen = tmp_path / "models/snapshots" / ("c" * 40)
    gemma = tmp_path / "models/snapshots" / ("d" * 40)
    source.mkdir()
    qwen.mkdir(parents=True)
    gemma.mkdir(parents=True)
    interpreter = Path(sys.executable).resolve(strict=True)
    probes = gate.Probes(
        git_head=lambda _path: "a" * 40,
        model_revision=lambda path: path.resolve().name,
        gpu_uuid=lambda: "GPU-v4-bind-test",
        file_sha256=gate._file_sha256,
        interpreter_version=gate._interpreter_version,
    )
    bound = binder.bind_manifest(
        manifest=manifest,
        source_root=source,
        model_paths={"qwen3": qwen, "gemma4": gemma},
        interpreter_path=interpreter,
        probes=probes,
    )
    for job in bound["jobs"]:
        identity = job["runtime_identity"]
        assert identity["source_commit"] == "a" * 40
        assert identity["gpu_uuid"] == "GPU-v4-bind-test"
        assert "deployment_receipt_sha256" not in identity


def test_v4_runtime_is_self_owned_and_v3_terminal_is_immutable():
    for path in STAGE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "rist_v3" not in text
        assert "RIST-C0-v3" not in text
        assert '"python3"' not in text
    report = V3 / "stages/C0_RESOLUTION/CALIBRATION_TERMINAL_REPORT_20260810.md"
    text = report.read_text(encoding="utf-8")
    assert "KILL_CURRENT_C0_V3_1_TRANSPORT_DO_NOT_OPEN_QUALIFICATION" in text
    assert "common high cells: none" in text

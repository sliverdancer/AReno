"""Finalize terminal RIST-E0-v1.1 preflight-invalid evidence."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path("/root/autodl-tmp/rist_e0_05da707")
SOURCE = ROOT / "source"
EVIDENCE = ROOT / "evidence"
VALIDATOR = SOURCE / (
    "research/reward_identifiability_supervision_topology/successors/"
    "rist_v1_1/stages/E0/validate_e0.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    preflight_path = EVIDENCE / "preflight_result.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("passed") is not False:
        raise RuntimeError("refusing terminal invalid finalization: preflight did not fail")
    if preflight.get("serving_started") is not False:
        raise RuntimeError("unexpected serving execution")

    spec = importlib.util.spec_from_file_location("rist_e0_validate", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen E0 validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stage_result = module.validate([], preflight_passed=False)
    if stage_result["decision"] != "INVALID_E0_PREFLIGHT_STOP":
        raise RuntimeError("unexpected validator decision")
    stage_result.update(
        {
            "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "source_commit": "05da707f19df590f613cd2403746495460809253",
            "preflight_result_sha256": sha256(preflight_path),
            "serving_started": False,
            "model_requests_sent": 0,
            "gpu_serving_seconds": 0,
            "checkpoints_downloaded_or_replaced": False,
            "qualification_or_heldout_data_opened": False,
            "terminal_reason": (
                "Frozen command `python -c \\\"import areno_accel\\\"` failed because "
                "the compiled extension is registered as `areno.accel._areno_accel`; "
                "the actual extension import and `areno check` both passed."
            ),
            "next_action": (
                "Freeze a new E0 protocol revision that checks the actual registered "
                "extension module; do not reinterpret or repair RIST-E0-v1.1."
            ),
        }
    )
    (EVIDENCE / "stage_result.json").write_text(
        json.dumps(stage_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    final_gpu = {
        "returncode": gpu.returncode,
        "stdout": gpu.stdout,
        "stderr": gpu.stderr,
    }
    (EVIDENCE / "final_gpu_processes.json").write_text(
        json.dumps(final_gpu, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if gpu.returncode != 0 or gpu.stdout.strip():
        raise RuntimeError("GPU process list was not clean at finalization")

    hook = {
        "schema_version": 1,
        "protocol_family": "RIST-v1.1",
        "phase": "E0_GPU_CANARY",
        "source_phase_decision": "INVALID_E0_PREFLIGHT_STOP",
        "decision": "STAY_DIAGNOSTIC_FREEZE_E0_V1_2",
        "upgraded": False,
        "main_conference_route_open": False,
        "reason": (
            "No serving request or scientific response was produced. The failure is a "
            "frozen preflight symbol mismatch, so it neither supports nor falsifies the "
            "reward-resolution or supervision-topology hypothesis."
        ),
        "required_next_gate": (
            "A separately frozen E0 revision must pass native serving for both model "
            "families before any P2.1 qualification protocol can open."
        ),
    }
    (EVIDENCE / "hook_result.json").write_text(
        json.dumps(hook, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = """# RIST-E0-v1.1 terminal report

Decision: `INVALID_E0_PREFLIGHT_STOP`

The frozen top-level import `import areno_accel` failed with
`ModuleNotFoundError`. The source registers the compiled extension as
`areno.accel._areno_accel`; that actual import passed from the isolated editable
checkout, and `areno check` reported ready. CUDA, exact source HEAD, source
cleanliness, and the empty GPU process list also passed.

Per the frozen protocol, this mismatch terminates v1.1 before serving. No model
was launched, no request was sent, no training occurred, no qualification or
held-out data was opened, and no checkpoint was downloaded or replaced. GPU
serving time was zero seconds.

This is infrastructure-only evidence. It cannot update the scientific or
main-conference claim. The next valid action is to freeze E0-v1.2 with the
registered extension import and rerun the independent canary under a new GPU
authorization.
"""
    (EVIDENCE / "TERMINAL_REPORT.md").write_text(report, encoding="utf-8")

    manifest = {
        path.name: sha256(path)
        for path in sorted(EVIDENCE.iterdir())
        if path.is_file() and path.name != "evidence_sha256.json"
    }
    (EVIDENCE / "evidence_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(stage_result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

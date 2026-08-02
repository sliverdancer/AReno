"""Audit and archive terminal RIST-P2.1-v1.0 evidence."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/rist_p2_1_0f1024b")
EVIDENCE = ROOT / "evidence"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


def main() -> int:
    stage = json.loads((EVIDENCE / "stage_result.json").read_text(encoding="utf-8"))
    qwen = json.loads((EVIDENCE / "qwen3_0_6b_result.json").read_text(encoding="utf-8"))
    gemma = json.loads((EVIDENCE / "gemma4_e2b_it_result.json").read_text(encoding="utf-8"))
    timeline = json.loads((EVIDENCE / "execution_timeline.json").read_text(encoding="utf-8"))
    qwen_log = (EVIDENCE / "qwen3_0_6b_server.log").read_text(
        encoding="utf-8", errors="replace"
    )
    gemma_log = (EVIDENCE / "gemma4_e2b_it_server.log").read_text(
        encoding="utf-8", errors="replace"
    )
    qwen_model = next(
        model for model in stage["models"] if model["model_cell"] == "qwen3_0_6b"
    )
    gemma_model = next(
        model for model in stage["models"] if model["model_cell"] == "gemma4_e2b_it"
    )
    qwen_raw = sum(
        len(trajectory.get("raw_responses", [])) for trajectory in qwen["trajectories"]
    )
    qwen_http_200 = count(r'POST /v1/chat/completions HTTP/1\.1" 200 OK', qwen_log)
    qwen_http_5xx = count(r'POST /v1/chat/completions HTTP/1\.1" 5\d\d', qwen_log)
    gemma_http_200 = count(r'POST /v1/chat/completions HTTP/1\.1" 200 OK', gemma_log)
    gemma_http_5xx = count(r'POST /v1/chat/completions HTTP/1\.1" 5\d\d', gemma_log)
    oom_detected = "torch.OutOfMemoryError" in gemma_log and "Tried to allocate 338.00 MiB" in gemma_log
    qwen_strata_meeting_gate = sum(
        int(value) >= 2 for value in qwen_model["stratum_mixed_groups"].values()
    )
    pass_condition_unreachable = qwen_strata_meeting_gate < 2
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
    relevant = subprocess.run(
        [
            "bash",
            "-c",
            "ps -eo pid,cmd | grep -E 'areno (serve|train)|run_p2|uvicorn' | grep -v grep || true",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    gpu_clean = gpu.returncode == 0 and not gpu.stdout.strip()
    relevant_clean = not relevant.stdout.strip()
    checks = {
        "formal_invalid_decision": stage.get("decision")
        == "INVALID_P2_1_PREFLIGHT_OR_INFRASTRUCTURE",
        "qwen_complete_256": len(qwen["trajectories"]) == 256,
        "qwen_raw_1024": qwen_raw == 1024,
        "qwen_http_200_1024": qwen_http_200 == 1024,
        "qwen_http_5xx_zero": qwen_http_5xx == 0,
        "qwen_all_per_model_gates_pass": qwen_model.get("passed") is True,
        "qwen_cross_stratum_gate_unreachable": pass_condition_unreachable,
        "gemma_retained_trajectories_zero": len(gemma["trajectories"]) == 0,
        "gemma_http_200_partial_responses": gemma_http_200 == 3,
        "gemma_http_500_one": gemma_http_5xx == 1,
        "gemma_oom_detected": oom_detected,
        "gemma_partial_raw_evidence_lost": gemma_http_200 > 0
        and len(gemma["trajectories"]) == 0,
        "retry_count_zero": int(qwen.get("retry_count", -1)) == 0
        and int(gemma.get("retry_count", -1)) == 0,
        "fabricated_call_count_zero": int(qwen.get("fabricated_call_count", -1)) == 0
        and int(gemma.get("fabricated_call_count", -1)) == 0,
        "within_gpu_limit": float(timeline["gpu_serving_elapsed_seconds"]) <= 18000.0,
        "training_not_performed": timeline.get("training_performed") is False,
        "heldout_not_opened": timeline.get("heldout_data_opened") is False,
        "checkpoint_unchanged": timeline.get("checkpoint_download_or_replacement") is False,
        "final_gpu_clean": gpu_clean,
        "final_relevant_processes_clean": relevant_clean,
    }
    audit = {
        "schema_version": 1,
        "protocol": "RIST-P2.1-v1.0",
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "formal_stage_status": stage.get("stage_status"),
        "formal_decision": stage.get("decision"),
        "audit_complete": all(checks.values()),
        "checks": checks,
        "gpu_serving_elapsed_seconds": float(timeline["gpu_serving_elapsed_seconds"]),
        "gpu_time_limit_seconds": 18000.0,
        "qwen": {
            "trajectory_count": len(qwen["trajectories"]),
            "raw_response_count": qwen_raw,
            "http_200_count": qwen_http_200,
            "http_5xx_count": qwen_http_5xx,
            "analysis": qwen_model,
            "strata_meeting_two_mixed_group_gate": qwen_strata_meeting_gate,
        },
        "gemma": {
            "trajectory_count": len(gemma["trajectories"]),
            "retained_raw_response_count": sum(
                len(trajectory.get("raw_responses", []))
                for trajectory in gemma["trajectories"]
            ),
            "http_200_count": gemma_http_200,
            "http_5xx_count": gemma_http_5xx,
            "infrastructure_error": gemma.get("infrastructure_error"),
            "oom_detected": oom_detected,
            "partial_response_evidence_loss": gemma_http_200 > 0
            and len(gemma["trajectories"]) == 0,
            "analysis": gemma_model,
        },
        "frozen_pass_condition_unreachable_from_complete_qwen_cell": (
            pass_condition_unreachable
        ),
        "unchanged_infrastructure_rerun_has_decision_value": False,
        "scientific_interpretation": (
            "CROSS_FAMILY_NOT_ESTIMABLE; COMPLETE_QWEN_CELL_IS_DIAGNOSTIC_ONLY"
        ),
        "training_performed": False,
        "heldout_data_opened": False,
    }
    if not audit["audit_complete"]:
        raise RuntimeError(f"P2.1 evidence audit failed: {audit}")
    (EVIDENCE / "audit_result.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hook = {
        "schema_version": 1,
        "protocol_family": "RIST-v1.1",
        "phase": "P2_1_GPU_QUALIFICATION",
        "source_phase_decision": "INVALID_P2_1_PREFLIGHT_OR_INFRASTRUCTURE",
        "formal_stage_disposition": "INVALID_PROTOCOL_STOP",
        "decision": "CLOSE_CURRENT_RIST_V1_1_ROUTE_NO_UNCHANGED_RERUN_VALUE",
        "upgraded": False,
        "main_conference_route_open": False,
        "reason": (
            "Gemma OOM and partial-response evidence loss make cross-family evidence invalid. "
            "Separately, the complete Qwen cell already fails the frozen two-strata mixed-group "
            "condition, so repairing Gemma alone cannot make this protocol pass."
        ),
        "claim_boundary": (
            "Do not claim the research hypothesis is falsified and do not splice Qwen into a "
            "successor. The defensible conclusion is that this frozen instrument and route cannot advance."
        ),
        "next_action": (
            "Archive and stop training escalation. Any continuation must be a newly designed "
            "instrument with long-prompt memory canaries and incremental raw-response journaling."
        ),
    }
    (EVIDENCE / "hook_result.json").write_text(
        json.dumps(hook, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# RIST-P2.1-v1.0 terminal report

Formal decision: `INVALID_P2_1_PREFLIGHT_OR_INFRASTRUCTURE`

Qwen completed all 256 trajectories and retained all 1,024 raw responses. It
passed every per-model interface and reward-resolution gate: first-turn
executable rate 0.9922, four-turn completion 0.9023, strict success 0.8086,
7/32 mixed groups, 56/256 non-zero-advantage trajectories, and zero retries or
fabricated calls.

The Qwen mixed groups were distributed as low=6 and intermediate=1. Therefore
only one stratum had at least two mixed groups, while the frozen cross-family
gate requires at least two such strata for every checkpoint. The P2.1 PASS
condition is already unreachable regardless of a future Gemma result.

Gemma passed health but failed during the first trajectory with CUDA OOM: the
worker attempted an additional 338 MiB with about 81 MiB free. The server log
records three HTTP 200 scientific responses followed by one HTTP 500, but the
trajectory-transactional client retained zero partial raw responses. This is
both an infrastructure failure and evidence loss. No retry was performed.

The serving window was {float(timeline['gpu_serving_elapsed_seconds']):.3f}
seconds of the 18,000-second limit. No training occurred, held-out remained
unopened, checkpoints were not downloaded or replaced, and final GPU and
relevant-process listings were empty.

Because the formal stage is invalid, cross-family reward resolution is not
estimable and Qwen cannot be spliced into another revision. Because Qwen also
makes the frozen PASS gate unattainable, an unchanged Gemma-only infrastructure
rerun has no decision value. The current RIST-v1.1 route is closed without
opening P3 training. This does not falsify the broad supervision-topology
hypothesis; it rejects this instrument and execution route.
"""
    (EVIDENCE / "TERMINAL_REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        path.name: sha256(path)
        for path in sorted(EVIDENCE.iterdir())
        if path.is_file()
        and path.name not in {"evidence_sha256.json", "rist_p2_1_evidence.tgz"}
    }
    (EVIDENCE / "evidence_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

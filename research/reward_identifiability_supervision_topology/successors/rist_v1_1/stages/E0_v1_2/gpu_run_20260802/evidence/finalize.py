"""Independently audit and archive terminal RIST-E0-v1.2 evidence."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/rist_e0_v1_2_511ed73")
EVIDENCE = ROOT / "evidence"
CELLS = ("qwen3_0_6b", "gemma4_e2b_it")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_audit(cell: str) -> dict[str, Any]:
    payload = json.loads((EVIDENCE / f"{cell}_result.json").read_text(encoding="utf-8"))
    trajectories = payload.get("trajectories") or []
    records = [record for trajectory in trajectories for record in trajectory.get("records", [])]
    raw_responses = [raw for trajectory in trajectories for raw in trajectory.get("raw_responses", [])]
    server_log = (EVIDENCE / f"{cell}_server.log").read_text(encoding="utf-8", errors="replace")
    summary = {
        "model_cell": cell,
        "trajectory_count": len(trajectories),
        "record_count": len(records),
        "raw_response_count": len(raw_responses),
        "declared_raw_response_count": sum(
            int(trajectory.get("raw_response_count", 0)) for trajectory in trajectories
        ),
        "parse_valid_count": sum(bool(record.get("parse_valid")) for record in records),
        "exact_instruction_count": sum(
            bool(record.get("exact_instruction")) for record in records
        ),
        "http_200_chat_completion_count": len(
            re.findall(r'POST /v1/chat/completions HTTP/1\.1" 200 OK', server_log)
        ),
        "health_200_count": len(re.findall(r'GET /health HTTP/1\.1" 200 OK', server_log)),
        "http_5xx_count": len(re.findall(r'HTTP/1\.1" 5\d\d', server_log)),
        "infrastructure_error": payload.get("infrastructure_error"),
        "fabricated_call_count": int(payload.get("fabricated_call_count", -1)),
        "retry_count": int(payload.get("retry_count", -1)),
        "post_request_shutdown_warning": "Application shutdown failed" in server_log,
    }
    summary["passed"] = all(
        [
            summary["trajectory_count"] == 2,
            summary["record_count"] == 8,
            summary["raw_response_count"] == 8,
            summary["declared_raw_response_count"] == 8,
            summary["parse_valid_count"] == 8,
            summary["exact_instruction_count"] == 8,
            summary["http_200_chat_completion_count"] == 8,
            summary["health_200_count"] >= 1,
            summary["http_5xx_count"] == 0,
            summary["infrastructure_error"] is None,
            summary["fabricated_call_count"] == 0,
            summary["retry_count"] == 0,
        ]
    )
    return summary


def main() -> int:
    stage_result = json.loads((EVIDENCE / "stage_result.json").read_text(encoding="utf-8"))
    timeline = json.loads((EVIDENCE / "execution_timeline.json").read_text(encoding="utf-8"))
    models = [model_audit(cell) for cell in CELLS]
    process_check = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    gpu_clean = process_check.returncode == 0 and not process_check.stdout.strip()
    relevant = subprocess.run(
        [
            "bash",
            "-c",
            "ps -eo pid,cmd | grep -E 'areno (serve|train)|run_e0_canary|uvicorn' | grep -v grep || true",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    relevant_clean = not relevant.stdout.strip()
    elapsed = float(timeline.get("gpu_serving_elapsed_seconds", float("inf")))
    passed = all(
        [
            stage_result.get("decision")
            == "PASS_E0_INFRASTRUCTURE_TO_P2_1_PROTOCOL_FREEZE",
            stage_result.get("stage_status") == "PASS",
            all(model["passed"] for model in models),
            elapsed <= 1800.0,
            timeline.get("gpu_processes_final") == [],
            timeline.get("training_performed") is False,
            timeline.get("qualification_data_opened") is False,
            timeline.get("checkpoint_download_or_replacement") is False,
            gpu_clean,
            relevant_clean,
        ]
    )
    audit = {
        "schema_version": 1,
        "protocol": "RIST-E0-v1.2",
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "passed": passed,
        "decision": stage_result.get("decision"),
        "models": models,
        "gpu_serving_elapsed_seconds": elapsed,
        "gpu_time_limit_seconds": 1800.0,
        "gpu_processes_final_clean": gpu_clean,
        "relevant_processes_final_clean": relevant_clean,
        "training_performed": False,
        "qualification_or_heldout_data_opened": False,
        "checkpoint_downloaded_or_replaced": False,
        "post_request_cleanup_note": (
            "Both servers accepted SIGTERM after all eight responses. Worker teardown emitted "
            "an application-shutdown warning; final GPU/process checks were clean. This occurred "
            "after request collection and is outside the frozen zero-5xx/client-exception gate."
        ),
        "scientific_interpretation": "FORBIDDEN_INFRASTRUCTURE_ONLY",
    }
    if not passed:
        raise RuntimeError(f"independent E0 audit failed: {audit}")
    (EVIDENCE / "audit_result.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hook = {
        "schema_version": 1,
        "protocol_family": "RIST-v1.1",
        "phase": "E0_V1_2_GPU_CANARY",
        "source_phase_decision": "PASS_E0_INFRASTRUCTURE_TO_P2_1_PROTOCOL_FREEZE",
        "decision": "STAY_DIAGNOSTIC_FREEZE_P2_1_PROTOCOL",
        "upgraded": False,
        "main_conference_route_open": False,
        "reason": (
            "Both model families passed the independent infrastructure canary, but E0 contains "
            "explicit answers and no scientific reward-resolution or training interaction evidence."
        ),
        "required_next_gate": (
            "Freeze P2.1 qualification without exposing its rows to serving; request separate GPU "
            "authorization only after the protocol, task manifest, cross-family gates, and stop rules pass CPU review."
        ),
    }
    (EVIDENCE / "hook_result.json").write_text(
        json.dumps(hook, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# RIST-E0-v1.2 terminal report

Decision: `PASS_E0_INFRASTRUCTURE_TO_P2_1_PROTOCOL_FREEZE`

Both frozen native-backend serving cells passed independently. Qwen3-0.6B and
Gemma-4 E2B each produced 8/8 raw HTTP 200 responses, 8/8 parser-valid single
offered tool calls, and 8/8 exact instructed tool/code pairs, with zero retries,
fabricated calls, HTTP 5xx responses, or client infrastructure exceptions.

The sequential serving window was {elapsed:.3f} seconds against the frozen
1800-second ceiling. No training occurred, no qualification or held-out rows
were opened, no checkpoint was downloaded or replaced, and final GPU and
relevant-process listings were empty.

Both servers emitted an application-shutdown warning after all responses while
handling the requested SIGTERM because a worker had already received the same
signal. This post-request cleanup warning did not affect a response, produce a
5xx/client exception, leave a process behind, or alter the mechanical validator
decision; it is retained in the raw server logs and independent audit.

E0 is infrastructure-only and cannot update the scientific or main-conference
claim. The permitted next action is CPU freezing of P2.1; scientific serving
still requires a new authorization after that freeze.
"""
    (EVIDENCE / "TERMINAL_REPORT.md").write_text(report, encoding="utf-8")

    manifest = {
        path.name: sha256(path)
        for path in sorted(EVIDENCE.iterdir())
        if path.is_file()
        and path.name not in {"evidence_sha256.json", "rist_e0_v1_2_evidence.tgz"}
    }
    (EVIDENCE / "evidence_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

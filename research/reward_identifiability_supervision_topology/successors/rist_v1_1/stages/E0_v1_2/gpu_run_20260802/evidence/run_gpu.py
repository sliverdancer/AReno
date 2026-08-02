"""Execute frozen RIST-E0-v1.2 sequential serving under a hard GPU clock."""

from __future__ import annotations

import datetime as dt
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/rist_e0_v1_2_511ed73")
STAGE = ROOT / (
    "research/reward_identifiability_supervision_topology/successors/"
    "rist_v1_1/stages/E0_v1_2"
)
EVIDENCE = ROOT / "evidence"
PYTHON = ROOT / "venv/bin/python"
ARENO = ROOT / "venv/bin/areno"
RUNNER = STAGE / "run_e0_canary.py"
VALIDATOR = STAGE / "validate_e0.py"
TASKS = STAGE / "canary_tasks.json"
MANIFEST = STAGE / "EXECUTION_MANIFEST.json"
PREFLIGHT = EVIDENCE / "preflight_result.json"
GPU_LIMIT_SECONDS = 1800.0
MODELS = (
    (
        "qwen3_0_6b",
        Path("/root/autodl-tmp/modelscope-cache/models/Qwen--Qwen3-0.6B/snapshots/master"),
        "modelscope",
    ),
    (
        "gemma4_e2b_it",
        Path("/root/autodl-tmp/hf-cache/google--gemma-4-E2B-it"),
        "hf",
    ),
)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def gpu_processes() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def wait_health(process: subprocess.Popen[str], deadline: float) -> dict[str, str]:
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before health, returncode={process.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload == {"status": "ok"}:
                return payload
            last_error = f"unexpected health payload: {payload!r}"
        except Exception as exc:  # Health probing is allowed; model requests are never retried.
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0)
    raise RuntimeError(f"health deadline exceeded: {last_error}")


def stop_server(process: subprocess.Popen[str], cell: str) -> dict[str, Any]:
    record: dict[str, Any] = {"cell": cell, "stop_started_utc": utc_now()}
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=45)
            record["signal"] = "SIGTERM"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
            record["signal"] = "SIGKILL_AFTER_TIMEOUT"
    record["returncode"] = process.returncode
    cleanup_deadline = time.monotonic() + 60.0
    while time.monotonic() < cleanup_deadline:
        remaining = gpu_processes()
        if not remaining:
            record["gpu_processes_after_stop"] = []
            record["stop_completed_utc"] = utc_now()
            return record
        time.sleep(2.0)
    record["gpu_processes_after_stop"] = gpu_processes()
    raise RuntimeError(f"GPU processes remained after stopping {cell}: {record}")


def infrastructure_result(cell: str, error_type: str, error: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "RIST-E0-v1.2",
        "scientific_interpretation": "FORBIDDEN_INFRASTRUCTURE_ONLY",
        "model_cell": cell,
        "canary_tasks_sha256": (
            "c0a4f8d8fa4c15765b57702797f44fcf2bbc0e637b570141d439ad465fce6c72"
        ),
        "expected_trajectories": 2,
        "expected_raw_responses": 8,
        "elapsed_seconds": 0.0,
        "infrastructure_error": {"error_type": error_type, "error": error},
        "trajectories": [],
        "fabricated_call_count": 0,
        "retry_count": 0,
    }


def interface_summary(result: dict[str, Any]) -> dict[str, Any]:
    trajectories = result.get("trajectories") or []
    records = [record for trajectory in trajectories for record in trajectory.get("records", [])]
    return {
        "infrastructure_error": result.get("infrastructure_error"),
        "trajectory_count": len(trajectories),
        "raw_response_count": sum(int(t.get("raw_response_count", 0)) for t in trajectories),
        "record_count": len(records),
        "parse_valid_count": sum(bool(record.get("parse_valid")) for record in records),
        "exact_instruction_count": sum(bool(record.get("exact_instruction")) for record in records),
        "fabricated_call_count": int(result.get("fabricated_call_count", -1)),
        "retry_count": int(result.get("retry_count", -1)),
    }


def main() -> int:
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("protocol") != "RIST-E0-v1.2" or preflight.get("passed") is not True:
        raise RuntimeError("refusing GPU execution without a passing v1.2 preflight")
    if gpu_processes():
        raise RuntimeError("GPU process list is not clean immediately before serving")

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "PYTHONUNBUFFERED": "1",
        }
    )
    timeline: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "RIST-E0-v1.2",
        "gpu_limit_seconds": GPU_LIMIT_SECONDS,
        "training_performed": False,
        "qualification_data_opened": False,
        "checkpoint_download_or_replacement": False,
        "events": [],
    }
    results: list[Path] = []
    serving_started = time.monotonic()
    timeline["gpu_window_started_utc"] = utc_now()
    write_json(EVIDENCE / "execution_timeline.json", timeline)

    terminal_infrastructure_error: str | None = None
    for model_index, (cell, model_path, model_hub) in enumerate(MODELS):
        elapsed = time.monotonic() - serving_started
        if elapsed >= GPU_LIMIT_SECONDS - 180.0:
            terminal_infrastructure_error = "insufficient time remained under the frozen GPU ceiling"
            result_path = EVIDENCE / f"{cell}_result.json"
            write_json(
                result_path,
                infrastructure_result(cell, "GpuTimeLimit", terminal_infrastructure_error),
            )
            results.append(result_path)
            break

        command = [
            str(ARENO),
            "serve",
            "--model-path",
            str(model_path),
            "--model-hub",
            model_hub,
            "--tp-size",
            "1",
            "--world-size",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--max-running-prompts",
            "1",
            "--default-max-tokens",
            "96",
            "--eager-decode",
            "--attn-backend",
            "native",
            "--disable-thinking",
        ]
        event: dict[str, Any] = {
            "cell": cell,
            "server_command": command,
            "server_started_utc": utc_now(),
            "gpu_elapsed_at_start_seconds": time.monotonic() - serving_started,
        }
        timeline["events"].append(event)
        write_json(EVIDENCE / "execution_timeline.json", timeline)
        log_path = EVIDENCE / f"{cell}_server.log"
        with log_path.open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            event["server_pid"] = server.pid
            try:
                health_deadline = min(
                    time.monotonic() + 480.0,
                    serving_started + GPU_LIMIT_SECONDS - 120.0,
                )
                event["health"] = wait_health(server, health_deadline)
                event["health_passed_utc"] = utc_now()
                result_path = EVIDENCE / f"{cell}_result.json"
                remaining = serving_started + GPU_LIMIT_SECONDS - time.monotonic() - 90.0
                if remaining <= 0:
                    raise RuntimeError("GPU ceiling reached before canary requests")
                runner_command = [
                    str(PYTHON),
                    str(RUNNER),
                    "--base-url",
                    "http://127.0.0.1:8000/v1",
                    "--api-key",
                    "EMPTY",
                    "--model-cell",
                    cell,
                    "--tasks",
                    str(TASKS),
                    "--manifest",
                    str(MANIFEST),
                    "--output",
                    str(result_path),
                    "--timeout-seconds",
                    str(min(300.0, max(30.0, remaining))),
                ]
                event["runner_command"] = runner_command
                runner = subprocess.run(
                    runner_command,
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=remaining,
                    check=False,
                )
                event["runner_returncode"] = runner.returncode
                event["runner_stdout"] = runner.stdout
                event["runner_stderr"] = runner.stderr
                if not result_path.is_file():
                    write_json(
                        result_path,
                        infrastructure_result(
                            cell,
                            "RunnerDidNotWriteResult",
                            f"returncode={runner.returncode}: {runner.stderr[-1000:]}",
                        ),
                    )
                result = json.loads(result_path.read_text(encoding="utf-8"))
                event["interface_summary"] = interface_summary(result)
                results.append(result_path)
                if result.get("infrastructure_error") is not None:
                    terminal_infrastructure_error = (
                        f"{cell} runner infrastructure error: {result['infrastructure_error']}"
                    )
            except Exception as exc:
                terminal_infrastructure_error = f"{cell}: {type(exc).__name__}: {exc}"
                result_path = EVIDENCE / f"{cell}_result.json"
                if not result_path.is_file():
                    write_json(
                        result_path,
                        infrastructure_result(cell, type(exc).__name__, str(exc)),
                    )
                    results.append(result_path)
                event["exception"] = terminal_infrastructure_error
            finally:
                event["server_stop"] = stop_server(server, cell)
                event["gpu_elapsed_after_stop_seconds"] = time.monotonic() - serving_started
                write_json(EVIDENCE / "execution_timeline.json", timeline)

        if terminal_infrastructure_error is not None:
            for remaining_cell, _, _ in MODELS[model_index + 1 :]:
                result_path = EVIDENCE / f"{remaining_cell}_result.json"
                write_json(
                    result_path,
                    infrastructure_result(
                        remaining_cell,
                        "PriorInfrastructureStop",
                        terminal_infrastructure_error,
                    ),
                )
                results.append(result_path)
            break

    timeline["gpu_window_ended_utc"] = utc_now()
    timeline["gpu_serving_elapsed_seconds"] = time.monotonic() - serving_started
    timeline["gpu_processes_final"] = gpu_processes()
    write_json(EVIDENCE / "execution_timeline.json", timeline)
    if timeline["gpu_serving_elapsed_seconds"] > GPU_LIMIT_SECONDS:
        raise RuntimeError("frozen GPU time limit exceeded")
    if timeline["gpu_processes_final"]:
        raise RuntimeError("GPU process list not empty after execution")

    ordered_results = [EVIDENCE / f"{cell}_result.json" for cell, _, _ in MODELS]
    validator_command = [
        str(PYTHON),
        str(VALIDATOR),
        "--preflight-result",
        str(PREFLIGHT),
    ]
    for path in ordered_results:
        validator_command.extend(["--result", str(path)])
    validator_command.extend(["--output", str(EVIDENCE / "stage_result.json")])
    validator = subprocess.run(
        validator_command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    (EVIDENCE / "validator_stdout.txt").write_text(validator.stdout, encoding="utf-8")
    (EVIDENCE / "validator_stderr.txt").write_text(validator.stderr, encoding="utf-8")
    if not (EVIDENCE / "stage_result.json").is_file():
        raise RuntimeError(f"validator did not write stage result: {validator.stderr}")
    stage_result = json.loads((EVIDENCE / "stage_result.json").read_text(encoding="utf-8"))
    print(json.dumps(stage_result, indent=2, sort_keys=True))
    return 0 if stage_result["stage_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

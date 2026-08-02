"""Execute frozen RIST-P2.1-v1.0 sequential cross-family qualification."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/rist_p2_1_0f1024b")
STAGE = ROOT / (
    "research/reward_identifiability_supervision_topology/successors/"
    "rist_v1_1/stages/P2_1"
)
DATA = ROOT / "research/reward_identifiability_supervision_topology/stages/P1/data"
EVIDENCE = ROOT / "evidence"
PYTHON = ROOT / "venv/bin/python"
ARENO = ROOT / "venv/bin/areno"
CLIENT = STAGE / "run_p2_1_client.py"
ANALYZER = STAGE / "analyze_p2_1.py"
MANIFEST = STAGE / "EXECUTION_MANIFEST.json"
PREFLIGHT = EVIDENCE / "preflight_result.json"
GPU_LIMIT_SECONDS = 18000.0
QUALIFICATION_SHA = "8018137606e12da0f0096ac86f11312d94d631326965198783ebf9cecc94570f"
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


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
        except Exception as exc:
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
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
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
        "protocol": "RIST-P2.1-v1.0",
        "model_cell": cell,
        "qualification_sha256": QUALIFICATION_SHA,
        "elapsed_seconds": 0.0,
        "expected_trajectories": 256,
        "trajectory_count": 0,
        "infrastructure_error": {"error_type": error_type, "error": error},
        "trajectories": [],
        "fabricated_call_count": 0,
        "retry_count": 0,
        "qualification_data_opened": False,
        "training_performed": False,
    }


def load_analyzer():
    spec = importlib.util.spec_from_file_location("rist_p2_1_analyzer", ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen P2.1 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("protocol") != "RIST-P2.1-v1.0" or preflight.get("passed") is not True:
        raise RuntimeError("refusing P2.1 without passing frozen preflight")
    if gpu_processes():
        raise RuntimeError("GPU process list not clean immediately before serving")

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
        "protocol": "RIST-P2.1-v1.0",
        "gpu_limit_seconds": GPU_LIMIT_SECONDS,
        "qualification_data_opened": False,
        "heldout_data_opened": False,
        "training_performed": False,
        "checkpoint_download_or_replacement": False,
        "events": [],
    }
    serving_started = time.monotonic()
    timeline["gpu_window_started_utc"] = utc_now()
    write_json(EVIDENCE / "execution_timeline.json", timeline)
    analyzer = load_analyzer()
    terminal_error: str | None = None

    for model_index, (cell, model_path, model_hub) in enumerate(MODELS):
        elapsed = time.monotonic() - serving_started
        if elapsed >= GPU_LIMIT_SECONDS - 300.0:
            terminal_error = "insufficient time remained under frozen GPU ceiling"
            result_path = EVIDENCE / f"{cell}_result.json"
            write_json(result_path, infrastructure_result(cell, "GpuTimeLimit", terminal_error))
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
            "128",
            "--eager-decode",
            "--attn-backend",
            "native",
            "--disable-thinking",
        ]
        event: dict[str, Any] = {
            "cell": cell,
            "server_command": command,
            "server_started_utc": utc_now(),
            "gpu_elapsed_at_start_seconds": elapsed,
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
                    serving_started + GPU_LIMIT_SECONDS - 300.0,
                )
                event["health"] = wait_health(server, health_deadline)
                event["health_passed_utc"] = utc_now()
                timeline["qualification_data_opened"] = True
                event["qualification_opened_utc"] = utc_now()
                write_json(EVIDENCE / "execution_timeline.json", timeline)
                result_path = EVIDENCE / f"{cell}_result.json"
                remaining = serving_started + GPU_LIMIT_SECONDS - time.monotonic() - 180.0
                if remaining <= 0:
                    raise RuntimeError("GPU ceiling reached before qualification client")
                client_command = [
                    str(PYTHON),
                    str(CLIENT),
                    "--base-url",
                    "http://127.0.0.1:8000/v1",
                    "--api-key",
                    "EMPTY",
                    "--model-cell",
                    cell,
                    "--data-dir",
                    str(DATA),
                    "--manifest",
                    str(MANIFEST),
                    "--output",
                    str(result_path),
                    "--timeout-seconds",
                    "300",
                ]
                event["client_command"] = client_command
                write_json(EVIDENCE / "execution_timeline.json", timeline)
                client = subprocess.run(
                    client_command,
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=remaining,
                    check=False,
                )
                event["client_returncode"] = client.returncode
                event["client_stdout"] = client.stdout
                event["client_stderr"] = client.stderr
                if not result_path.is_file():
                    write_json(
                        result_path,
                        infrastructure_result(
                            cell,
                            "ClientDidNotWriteResult",
                            f"returncode={client.returncode}: {client.stderr[-1000:]}",
                        ),
                    )
                result = json.loads(result_path.read_text(encoding="utf-8"))
                event["model_analysis"] = analyzer.analyze_model(result)
                if result.get("infrastructure_error") is not None:
                    terminal_error = (
                        f"{cell} client infrastructure error: {result['infrastructure_error']}"
                    )
            except Exception as exc:
                terminal_error = f"{cell}: {type(exc).__name__}: {exc}"
                result_path = EVIDENCE / f"{cell}_result.json"
                if not result_path.is_file():
                    write_json(
                        result_path,
                        infrastructure_result(cell, type(exc).__name__, str(exc)),
                    )
                event["exception"] = terminal_error
            finally:
                event["server_stop"] = stop_server(server, cell)
                event["gpu_elapsed_after_stop_seconds"] = time.monotonic() - serving_started
                write_json(EVIDENCE / "execution_timeline.json", timeline)

        if terminal_error is not None:
            for remaining_cell, _, _ in MODELS[model_index + 1 :]:
                result_path = EVIDENCE / f"{remaining_cell}_result.json"
                write_json(
                    result_path,
                    infrastructure_result(
                        remaining_cell, "PriorInfrastructureStop", terminal_error
                    ),
                )
            break

    timeline["gpu_window_ended_utc"] = utc_now()
    timeline["gpu_serving_elapsed_seconds"] = time.monotonic() - serving_started
    timeline["gpu_processes_final"] = gpu_processes()
    write_json(EVIDENCE / "execution_timeline.json", timeline)
    if timeline["gpu_serving_elapsed_seconds"] > GPU_LIMIT_SECONDS:
        raise RuntimeError("frozen five-hour GPU limit exceeded")
    if timeline["gpu_processes_final"]:
        raise RuntimeError("GPU processes remained after execution")

    result_paths = [EVIDENCE / f"{cell}_result.json" for cell, _, _ in MODELS]
    analyzer_command = [str(PYTHON), str(ANALYZER)]
    for result_path in result_paths:
        analyzer_command.extend(["--model-result", str(result_path)])
    analyzer_command.extend(["--output", str(EVIDENCE / "stage_result.json")])
    analysis = subprocess.run(
        analyzer_command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    (EVIDENCE / "analyzer_stdout.txt").write_text(analysis.stdout, encoding="utf-8")
    (EVIDENCE / "analyzer_stderr.txt").write_text(analysis.stderr, encoding="utf-8")
    if not (EVIDENCE / "stage_result.json").is_file():
        raise RuntimeError(f"analyzer did not write stage result: {analysis.stderr}")
    stage_result = json.loads((EVIDENCE / "stage_result.json").read_text(encoding="utf-8"))
    print(json.dumps(stage_result, indent=2, sort_keys=True))
    return 0 if stage_result["stage_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

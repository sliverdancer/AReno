"""Bounded remote controller for the frozen SAS-TR-v2.0 B2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from analyze_b2 import aggregate

FROZEN_SEEDS = (3101, 3202)
CELLS = {
    "D128": {"thinking": "default", "max_new_tokens": 128},
    "D512": {"thinking": "default", "max_new_tokens": 512},
    "N128": {"thinking": "disabled", "max_new_tokens": 128},
    "N512": {"thinking": "disabled", "max_new_tokens": 512},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gpu_snapshot() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_inputs(repo_root: Path, model_path: Path) -> dict[str, Any]:
    b1 = repo_root / "research" / "structured_action_supervision_v2" / "stages" / "B1"
    manifest_path = b1 / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in ("calibration", "validation", "reserve"):
        path = b1 / manifest["files"][split]["path"]
        if _sha256(path) != manifest["files"][split]["sha256"]:
            raise RuntimeError(f"frozen {split} hash mismatch")
    weights = sorted(model_path.glob("*.safetensors"))
    if len(weights) != 1:
        raise RuntimeError(f"expected one safetensors weight file, found {len(weights)}")
    weight_hash = _sha256(weights[0])
    if weight_hash != manifest["model"]["weights_sha256"]:
        raise RuntimeError("model weight hash does not match the frozen B1 manifest")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "weight_file": str(weights[0]),
        "weight_sha256": weight_hash,
    }


def _wait_health(base_url: str, process: subprocess.Popen[Any], deadline: float) -> None:
    health = base_url.rstrip("/") + "/health"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"AReno serving exited during startup with code {process.returncode}")
        try:
            with urllib.request.urlopen(health, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("AReno serving did not become healthy before the GPU deadline")


def _stop_server(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _run_client(
    *,
    script: Path,
    base_url: str,
    model: str,
    repo_root: Path,
    dataset: Path,
    output: Path,
    cell_id: str,
    split: str,
    max_new_tokens: int,
    seed: int,
    deadline: float,
    workers: int,
) -> None:
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("GPU deadline reached before client launch")
    command = [
        sys.executable,
        str(script),
        "--base-url",
        base_url,
        "--model",
        model,
        "--repo-root",
        str(repo_root),
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--cell-id",
        cell_id,
        "--split",
        split,
        "--max-new-tokens",
        str(max_new_tokens),
        "--sampling-seed",
        str(seed),
        "--deadline-epoch",
        str(deadline),
        "--workers",
        str(workers),
    ]
    subprocess.run(command, check=True, timeout=remaining)


def _serve(
    *,
    areno_bin: Path,
    model_path: Path,
    base_host: str,
    port: int,
    disabled: bool,
    log_path: Path,
) -> tuple[subprocess.Popen[Any], Any]:
    command = [
        str(areno_bin),
        "serve",
        "--model-path",
        str(model_path),
        "--tp-size",
        "1",
        "--world-size",
        "1",
        "--host",
        base_host,
        "--port",
        str(port),
        "--max-running-prompts",
        "8",
        "--default-max-tokens",
        "512",
        "--attn-backend",
        "native",
    ]
    if disabled:
        command.append("--disable-thinking")
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    return process, log_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--areno-bin", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--gpu-limit-s", type=int, default=3600)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.gpu_limit_s <= 3600:
        raise ValueError("gpu-limit-s must be between 1 and the authorized 3600 seconds")
    repo_root = args.repo_root.resolve()
    model_path = args.model_path.resolve()
    runtime = args.runtime_dir.resolve()
    if runtime.exists() and any(runtime.iterdir()):
        raise RuntimeError(f"refusing to reuse non-empty runtime directory: {runtime}")
    runtime.mkdir(parents=True, exist_ok=True)
    verification = _verify_inputs(repo_root, model_path)
    gpu_before = _gpu_snapshot()
    used_memory = int(gpu_before.split(",")[2].strip().split()[0])
    if used_memory > 64:
        raise RuntimeError(f"GPU is not idle before B2: {gpu_before}")

    root = repo_root / "research" / "structured_action_supervision_v2"
    b1 = root / "stages" / "B1"
    client = root / "run_b2_client.py"
    base_url = f"http://{args.host}:{args.port}"
    controller_started = time.time()
    deadline = controller_started + args.gpu_limit_s
    controller: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "SAS-TR-v2.0",
        "source_commit": args.source_commit,
        "gpu_limit_s": args.gpu_limit_s,
        "gpu_before": gpu_before,
        "started_epoch": controller_started,
        "verification": verification,
        "runs": [],
        "status": "RUNNING",
    }
    _json_write(runtime / "controller_result.json", controller)
    server: subprocess.Popen[Any] | None = None
    log_file: Any = None
    try:
        for mode in ("default", "disabled"):
            server, log_file = _serve(
                areno_bin=args.areno_bin.resolve(),
                model_path=model_path,
                base_host=args.host,
                port=args.port,
                disabled=mode == "disabled",
                log_path=runtime / f"serve_{mode}.log",
            )
            _wait_health(base_url, server, deadline)
            for cell_id, cell in CELLS.items():
                if cell["thinking"] != mode:
                    continue
                for seed in FROZEN_SEEDS:
                    output = runtime / f"calibration_{cell_id}_seed_{seed}.json"
                    _run_client(
                        script=client,
                        base_url=base_url,
                        model="policy",
                        repo_root=repo_root,
                        dataset=b1 / "calibration.jsonl",
                        output=output,
                        cell_id=cell_id,
                        split="calibration",
                        max_new_tokens=int(cell["max_new_tokens"]),
                        seed=seed,
                        deadline=deadline,
                        workers=args.workers,
                    )
                    controller["runs"].append({"cell_id": cell_id, "seed": seed, "split": "calibration", "path": str(output)})
                    _json_write(runtime / "controller_result.json", controller)
            if mode == "default":
                _stop_server(server)
                server = None
                log_file.close()
                log_file = None

        calibration_paths = sorted(runtime.glob("calibration_*.json"))
        calibration = aggregate(calibration_paths, "calibration")
        _json_write(runtime / "calibration_analysis.json", calibration)
        selected = calibration["selected_cell"]
        validation = None
        if selected is not None:
            for seed in FROZEN_SEEDS:
                output = runtime / f"validation_{selected}_seed_{seed}.json"
                _run_client(
                    script=client,
                    base_url=base_url,
                    model="policy",
                    repo_root=repo_root,
                    dataset=b1 / "validation.jsonl",
                    output=output,
                    cell_id=selected,
                    split="validation",
                    max_new_tokens=int(CELLS[selected]["max_new_tokens"]),
                    seed=seed,
                    deadline=deadline,
                    workers=args.workers,
                )
                controller["runs"].append({"cell_id": selected, "seed": seed, "split": "validation", "path": str(output)})
                _json_write(runtime / "controller_result.json", controller)
            validation = aggregate(sorted(runtime.glob("validation_*.json")), "validation")
            _json_write(runtime / "validation_analysis.json", validation)
            selected_metrics = validation["cell_results"][selected]
            decision = (
                f"PASS_B2_INTERFACE_{selected}"
                if selected_metrics["passes_frozen_gates"]
                else "KILL_QWEN3_0_6B_INSTRUMENT_VALIDATION_FAILURE"
            )
        else:
            decision = "KILL_QWEN3_0_6B_INSTRUMENT"
        controller.update(
            {
                "status": "COMPLETE",
                "decision": decision,
                "selected_cell": selected,
                "finished_epoch": time.time(),
                "elapsed_gpu_wall_s": time.time() - controller_started,
            }
        )
        _json_write(runtime / "controller_result.json", controller)
        print(json.dumps({"decision": decision, "selected_cell": selected}, sort_keys=True))
        return 0
    except BaseException as exc:
        controller.update(
            {
                "status": "ABORTED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "finished_epoch": time.time(),
                "elapsed_gpu_wall_s": time.time() - controller_started,
            }
        )
        _json_write(runtime / "controller_result.json", controller)
        raise
    finally:
        _stop_server(server)
        if log_file is not None:
            log_file.close()
        try:
            controller["gpu_after"] = _gpu_snapshot()
            _json_write(runtime / "controller_result.json", controller)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

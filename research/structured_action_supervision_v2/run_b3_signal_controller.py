"""Execute the bounded SAS-B3-v3.0 within-group signal preflight."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from analyze_b3_signal import analyze
from run_b2_client import _load_game, run_trajectory
from run_b2_controller import (
    _gpu_snapshot,
    _json_write,
    _preflight,
    _serve,
    _stop_server,
    _wait_health,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(repo_root: Path, model_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    b2_result_path = (
        repo_root
        / "research/structured_action_supervision_v2/stages/B2_1/stage_result.json"
    )
    if _sha256(b2_result_path) != manifest["parent"]["stage_result_sha256"]:
        raise RuntimeError("B2.1 parent stage-result hash mismatch")
    b2_result = json.loads(b2_result_path.read_text(encoding="utf-8"))
    if b2_result.get("decision") != manifest["parent"]["decision"]:
        raise RuntimeError("B2.1 did not qualify N128")
    dataset = repo_root / manifest["dataset"]["path"]
    if _sha256(dataset) != manifest["dataset"]["sha256"]:
        raise RuntimeError("B3-A dataset hash mismatch")
    weights = sorted(model_path.glob("*.safetensors"))
    if len(weights) != 1 or _sha256(weights[0]) != manifest["model"]["weights_sha256"]:
        raise RuntimeError("B3-A model weight hash mismatch")
    return {
        "b2_stage_result": str(b2_result_path),
        "b2_stage_result_sha256": _sha256(b2_result_path),
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "weight_file": str(weights[0]),
        "weight_sha256": _sha256(weights[0]),
    }


def _run_seed(
    *,
    rows: list[dict[str, Any]],
    game: Any,
    base_url: str,
    model: str,
    seed: int,
    deadline: float,
    output: Path,
    protocol_id: str,
) -> None:
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        futures = [
            pool.submit(
                run_trajectory,
                row=row,
                row_index=index,
                game=game,
                endpoint=endpoint,
                model=model,
                max_new_tokens=128,
                sampling_seed=seed,
                timeout=300.0,
                deadline_epoch=deadline,
            )
            for index, row in enumerate(rows)
        ]
        trajectories = [future.result() for future in futures]
    trajectories.sort(key=lambda item: item["row_index"])
    payload = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "stage": "B3-A",
        "sampling_seed": seed,
        "sampling_policy": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
        "started_epoch": started,
        "finished_epoch": time.time(),
        "trajectories": trajectories,
    }
    _json_write(output, payload)


def _signal_exit(signum: int, _frame: Any) -> None:
    raise SystemExit(f"B3-A controller received signal {signum}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--areno-bin", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpu-limit-s", type=int, default=1800)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.gpu_limit_s <= 1800:
        raise ValueError("gpu-limit-s must be between 1 and 1800 seconds")
    repo_root = args.repo_root.resolve()
    runtime = args.runtime_dir.resolve()
    if runtime.exists() and any(runtime.iterdir()):
        raise RuntimeError(f"refusing to reuse non-empty runtime directory: {runtime}")
    runtime.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        repo_root
        / "research/structured_action_supervision_v2/stages/B3/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = _verify(repo_root, args.model_path.resolve(), manifest)
    gpu_before = _gpu_snapshot()
    if int(gpu_before.split(",")[2].strip().split()[0]) > 64:
        raise RuntimeError(f"GPU is not idle before B3-A: {gpu_before}")
    started = time.time()
    deadline = started + args.gpu_limit_s
    controller: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": manifest["protocol_id"],
        "stage": manifest["stage"],
        "source_commit": args.source_commit,
        "gpu_limit_s": args.gpu_limit_s,
        "gpu_before": gpu_before,
        "started_epoch": started,
        "verification": verification,
        "runs": [],
        "status": "RUNNING",
    }
    _json_write(runtime / "controller_result.json", controller)
    server = None
    log_file = None
    signal.signal(signal.SIGINT, _signal_exit)
    signal.signal(signal.SIGTERM, _signal_exit)
    try:
        base_url = f"http://{args.host}:{args.port}"
        server, log_file = _serve(
            areno_bin=args.areno_bin.resolve(),
            model_path=args.model_path.resolve(),
            base_host=args.host,
            port=args.port,
            disabled=True,
            log_path=runtime / "serve_disabled.log",
            max_running_prompts=1,
        )
        _wait_health(base_url, server, deadline)
        _preflight(
            base_url=base_url,
            mode="disabled",
            output=runtime / "capacity_preflight.json",
            deadline=deadline,
        )
        dataset = repo_root / manifest["dataset"]["path"]
        rows = [
            json.loads(line)
            for line in dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        game = _load_game(repo_root)
        for seed in manifest["sampling_seeds"]:
            if time.time() >= deadline:
                raise TimeoutError("GPU deadline reached before all signal groups completed")
            output = runtime / f"signal_seed_{seed}.json"
            _run_seed(
                rows=rows,
                game=game,
                base_url=base_url,
                model="policy",
                seed=int(seed),
                deadline=deadline,
                output=output,
                protocol_id=manifest["protocol_id"],
            )
            controller["runs"].append({"seed": seed, "path": str(output)})
            _json_write(runtime / "controller_result.json", controller)
        analysis = analyze(sorted(runtime.glob("signal_seed_*.json")), manifest)
        _json_write(runtime / "signal_analysis.json", analysis)
        controller.update(
            {
                "status": "COMPLETE",
                "decision": analysis["decision"],
                "finished_epoch": time.time(),
                "elapsed_gpu_wall_s": time.time() - started,
            }
        )
        _json_write(runtime / "controller_result.json", controller)
        print(json.dumps({"decision": analysis["decision"]}, sort_keys=True))
        return 0
    except BaseException as exc:
        controller.update(
            {
                "status": "ABORTED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "finished_epoch": time.time(),
                "elapsed_gpu_wall_s": time.time() - started,
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

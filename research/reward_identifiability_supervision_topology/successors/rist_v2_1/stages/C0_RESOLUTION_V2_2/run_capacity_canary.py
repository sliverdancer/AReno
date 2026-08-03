"""Run an outcome-free eight-request serving-capacity canary for C0 v2.2."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[6]
EVALUATOR_PATH = Path(__file__).resolve().parents[1] / "D4_EVAL/evaluate_checkpoint.py"


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("rist_c0_v2_2_canary_http", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("strict HTTP client is unavailable")
    spec.loader.exec_module(module)
    return module


def _append(path: Path, row: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, (json.dumps(row, sort_keys=True) + "\n").encode())
    finally:
        os.close(descriptor)


def _request(task: dict[str, Any], seed: int) -> dict[str, Any]:
    turn = task["turns"][0]
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Perform the {name} registry operation.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        }
        for name in turn["offered_tools"]
    ]
    candidates = ", ".join(
        f"{row['label']}:{row['code']}" for row in turn["candidate_records"]
    )
    return {
        "model": "policy",
        "messages": [
            {
                "role": "system",
                "content": "Return exactly one offered tool call with one code argument.",
            },
            {
                "role": "user",
                "content": (
                    f"Offered tools: {', '.join(turn['offered_tools'])}. "
                    f"Target label: {turn['target_label']}. Candidates: {candidates}"
                ),
            },
        ],
        "tools": tools,
        "tool_choice": "required",
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 128,
        "seed": seed,
    }


def run_canary(
    pool_manifest: dict[str, Any],
    data_dir: Path,
    family: str,
    runtime_identity: dict[str, Any],
    journal_path: Path,
    post_json: Callable[[dict[str, Any]], dict[str, Any]],
    memory_used_mib: Callable[[], float],
) -> dict[str, Any]:
    if pool_manifest.get("protocol") != "RIST-C0-v2.2-FRESH-POOL":
        raise ValueError("unexpected C0 v2.2 pool manifest")
    if family not in {"qwen3", "gemma4"}:
        raise ValueError("capacity canary requires a frozen model family")
    if journal_path.exists():
        raise FileExistsError("capacity canary journal must be fresh")
    if runtime_identity.get("family") != family:
        raise ValueError("capacity canary runtime identity family mismatch")
    if float(runtime_identity.get("gpu_total_memory_gib", 0.0)) < 48.0:
        raise ValueError("C0 v2.2 capacity canary requires at least 48 GB GPU memory")
    split = pool_manifest["splits"]["capacity_canary"]
    source = data_dir / split["file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != split["sha256"]:
        raise ValueError("capacity canary source hash mismatch")
    tasks = [json.loads(line) for line in source.read_text().splitlines() if line]
    if len(tasks) != 1 or len(split["rollout_seeds"]) != 8:
        raise ValueError("capacity canary requires one task and eight frozen requests")
    start = time.monotonic()
    responses = {}
    infrastructure_error = None
    peak_memory_mib = float(memory_used_mib())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            int(seed): pool.submit(post_json, _request(tasks[0], int(seed)))
            for seed in split["rollout_seeds"]
        }
        while any(not future.done() for future in futures.values()):
            peak_memory_mib = max(peak_memory_mib, float(memory_used_mib()))
            time.sleep(0.05)
        for seed in split["rollout_seeds"]:
            try:
                response = futures[int(seed)].result()
                if not isinstance(response, dict) or not isinstance(response.get("choices"), list):
                    raise ValueError("capacity response lacks OpenAI choices")
                row = {
                    "request_seed": int(seed),
                    "response_sha256": hashlib.sha256(
                        json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "raw_response": response,
                }
                _append(journal_path, row)
                responses[int(seed)] = row["response_sha256"]
            except Exception as exc:
                infrastructure_error = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "request_seed": int(seed),
                }
                break
    complete = infrastructure_error is None and set(responses) == set(
        int(seed) for seed in split["rollout_seeds"]
    )
    return {
        "protocol": "RIST-C0-v2.2-CAPACITY-CANARY",
        "family": family,
        "runtime_identity": runtime_identity,
        "request_concurrency": 8,
        "expected_response_count": 8,
        "response_count": len(responses),
        "request_seeds": list(split["rollout_seeds"]),
        "peak_memory_mib": peak_memory_mib,
        "elapsed_seconds": time.monotonic() - start,
        "retry_count": 0,
        "infrastructure_error": infrastructure_error,
        "complete": complete,
        "outcomes_inspected": False,
        "scientific_result": False,
        "raw_journal_sha256": (
            hashlib.sha256(journal_path.read_bytes()).hexdigest()
            if journal_path.is_file()
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--family", choices=("qwen3", "gemma4"), required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluator = _load_evaluator()

    def memory_used_mib() -> float:
        import subprocess

        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        values = [float(line) for line in output.splitlines() if line.strip()]
        return sum(values)

    result = run_canary(
        json.loads(args.pool_manifest.read_text()),
        args.data_dir,
        args.family,
        json.loads(args.runtime_identity.read_text()),
        args.journal,
        lambda payload: evaluator._post_json(
            args.base_url, args.api_key, args.timeout_seconds, payload
        ),
        memory_used_mib,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

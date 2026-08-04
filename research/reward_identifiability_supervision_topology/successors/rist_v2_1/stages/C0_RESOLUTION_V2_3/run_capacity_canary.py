"""Run the outcome-free eight-request serving-capacity canary for C0 v2.3."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any, Callable


EVALUATOR_PATH = Path(__file__).resolve().parents[1] / "D4_EVAL/evaluate_checkpoint.py"
POOL_MANIFEST_SHA256 = "de183d3d4e3f4d375f14555ef69888dd909a7efb44580438bbceb59bbe8776fa"
CANARY_DATA_SHA256 = "4220775b278743d188857d02cf388dcaee18497e7630f8dae09cb17c782fec52"
BINDING_SHA256 = "aec17014d8cff07fd8f150130e57d9bcf69742dabb46c1e8196178ac54dbdc6f"
CANARY_SEEDS = tuple(range(18001, 18009))
RECEIPT_SHA256 = {
    "qwen3": "055269d1ab2ef7d5b02a416c27afaab6b039470a89889286cf8d91b820de5c90",
    "gemma4": "da49af33c36f26ff0c7030c7e929ef1ec0d358dc5dc51d3fbad41bafe21e83a6",
}
RECEIPT_PROTOCOL = "RIST-C0-v2.3-DEPLOYMENT-RECEIPT-v1"
IDENTITY_FIELDS = (
    "control_commit",
    "runtime_commit",
    "manifest_sha256",
    "model_revision",
    "gpu_uuid",
    "extension_sha256",
)


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("rist_c0_v2_3_canary_http", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("strict HTTP client is unavailable")
    spec.loader.exec_module(module)
    return module


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reserve_journal(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError("capacity canary journal must be fresh") from exc


def _append(descriptor: int, row: dict[str, Any]) -> None:
    os.write(descriptor, (json.dumps(row, sort_keys=True) + "\n").encode())
    os.fsync(descriptor)


def _validate_receipt(
    family: str,
    pool_manifest_bytes: bytes,
    binding_bytes: bytes,
    receipt_bytes: bytes,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if _sha256(pool_manifest_bytes) != POOL_MANIFEST_SHA256:
        raise ValueError("capacity canary pool manifest SHA mismatch")
    if _sha256(binding_bytes) != BINDING_SHA256:
        raise ValueError("capacity canary post-rent binding SHA mismatch")
    if _sha256(receipt_bytes) != RECEIPT_SHA256[family]:
        raise ValueError("capacity canary receipt artifact SHA mismatch")
    pool_manifest = json.loads(pool_manifest_bytes)
    binding = json.loads(binding_bytes)
    receipt = json.loads(receipt_bytes)
    if receipt.get("protocol") != RECEIPT_PROTOCOL:
        raise ValueError("capacity canary receipt protocol mismatch")
    body = {
        key: receipt[key]
        for key in (
            "protocol",
            "authority_sha256",
            "bindings",
            "manifest_base64",
            "launcher_command",
        )
    }
    if receipt.get("receipt_sha256") != _sha256(_canonical(body)):
        raise ValueError("capacity canary receipt self-SHA mismatch")
    receipt_bindings = receipt.get("bindings")
    if not isinstance(receipt_bindings, dict) or set(receipt_bindings) != set(IDENTITY_FIELDS):
        raise ValueError("capacity canary receipt identity schema mismatch")
    try:
        embedded_manifest = base64.b64decode(receipt["manifest_base64"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("capacity canary embedded manifest is invalid") from exc
    if embedded_manifest != pool_manifest_bytes:
        raise ValueError("capacity canary embedded manifest bytes mismatch")
    receipt_spec = binding.get("receipts", {}).get(family)
    if not isinstance(receipt_spec, dict):
        raise ValueError("capacity canary binding lacks family receipt")
    if (
        binding.get("protocol") != "RIST-C0-v2.3-POST-RENT-BINDING-v1"
        or binding.get("status")
        != "BOUND_BY_COMMIT_2a31e76b546dfbf9b5c801bc0efad3066254ec9f"
        or receipt_spec.get("artifact_sha256") != RECEIPT_SHA256[family]
        or receipt_spec.get("model_revision") != receipt_bindings["model_revision"]
        or binding.get("authority_canonical_sha256") != receipt.get("authority_sha256")
    ):
        raise ValueError("capacity canary binding and receipt mismatch")
    for field in ("control_commit", "runtime_commit", "manifest_sha256", "extension_sha256"):
        if binding.get(field) != receipt_bindings[field]:
            raise ValueError(f"capacity canary bound identity mismatch: {field}")
    if binding.get("gpu", {}).get("uuid") != receipt_bindings["gpu_uuid"]:
        raise ValueError("capacity canary bound identity mismatch: gpu_uuid")
    return pool_manifest, binding, receipt


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


def _expected_server_python(receipt: dict[str, Any]) -> str:
    candidates = [
        part
        for part in receipt.get("launcher_command", [])
        if isinstance(part, str) and part.endswith("/bin/python")
    ]
    if len(candidates) != 1:
        raise ValueError("capacity canary receipt must bind exactly one server Python")
    return candidates[0]


def run_canary(
    pool_manifest_bytes: bytes,
    data_dir: Path,
    family: str,
    runtime_identity: dict[str, Any],
    binding_bytes: bytes,
    receipt_bytes: bytes,
    journal_path: Path,
    post_json: Callable[[dict[str, Any]], dict[str, Any]],
    live_gpu_uuid: Callable[[], str],
    live_gpu_total_memory_mib: Callable[[], int],
    live_compute_process_names: Callable[[], list[str]],
    memory_used_mib: Callable[[], float],
) -> dict[str, Any]:
    if family not in RECEIPT_SHA256:
        raise ValueError("capacity canary requires a frozen model family")
    pool_manifest, binding, receipt = _validate_receipt(
        family, pool_manifest_bytes, binding_bytes, receipt_bytes
    )
    if pool_manifest.get("protocol") != "RIST-C0-v2.3-FRESH-POOL":
        raise ValueError("unexpected C0 v2.3 pool manifest")
    if runtime_identity.get("family") != family:
        raise ValueError("capacity canary runtime identity family mismatch")
    receipt_bindings = receipt["bindings"]
    for field in IDENTITY_FIELDS:
        if runtime_identity.get(field) != receipt_bindings[field]:
            raise ValueError(f"capacity canary runtime identity mismatch: {field}")
    observed_gpu_uuid = live_gpu_uuid()
    observed_memory_mib = int(live_gpu_total_memory_mib())
    if observed_gpu_uuid != receipt_bindings["gpu_uuid"]:
        raise ValueError("capacity canary live GPU UUID mismatch")
    if observed_memory_mib != int(binding["gpu"]["memory_total_mib"]):
        raise ValueError("capacity canary live GPU memory mismatch")
    if runtime_identity.get("gpu_total_memory_mib") != observed_memory_mib:
        raise ValueError("capacity canary runtime GPU memory mismatch")
    if observed_memory_mib < 48 * 1024:
        raise ValueError("C0 v2.3 capacity canary requires at least 48 GB GPU memory")
    split = pool_manifest["splits"]["capacity_canary"]
    source = data_dir / split["file"]
    if (
        split.get("sha256") != CANARY_DATA_SHA256
        or _sha256(source.read_bytes()) != CANARY_DATA_SHA256
    ):
        raise ValueError("capacity canary source hash mismatch")
    tasks = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    seeds = [int(seed) for seed in split["rollout_seeds"]]
    if len(tasks) != 1 or tuple(seeds) != CANARY_SEEDS:
        raise ValueError("capacity canary requires the exact eight frozen request seeds")

    expected_server_python = _expected_server_python(receipt)
    compute_processes = live_compute_process_names()
    if compute_processes != [expected_server_python]:
        raise ValueError("capacity canary requires an exclusive bound server process")

    descriptor = _reserve_journal(journal_path)
    start = time.monotonic()
    responses: dict[int, str] = {}
    infrastructure_error = None
    peak_memory_mib = float(memory_used_mib())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {seed: pool.submit(post_json, _request(tasks[0], seed)) for seed in seeds}
            while any(not future.done() for future in futures.values()):
                peak_memory_mib = max(peak_memory_mib, float(memory_used_mib()))
                time.sleep(0.05)
            for seed in seeds:
                try:
                    response = futures[seed].result()
                    if not isinstance(response, dict) or not isinstance(response.get("choices"), list):
                        raise ValueError("capacity response lacks OpenAI choices")
                    row = {
                        "request_seed": seed,
                        "response_sha256": _sha256(_canonical(response)),
                        "raw_response": response,
                    }
                    _append(descriptor, row)
                    responses[seed] = row["response_sha256"]
                except Exception as exc:
                    infrastructure_error = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "request_seed": seed,
                    }
                    break
    finally:
        os.close(descriptor)

    complete = infrastructure_error is None and set(responses) == set(seeds)
    return {
        "protocol": "RIST-C0-v2.3-CAPACITY-CANARY-v1",
        "family": family,
        "runtime_identity": runtime_identity,
        "request_concurrency": 8,
        "expected_response_count": 8,
        "response_count": len(responses),
        "request_seeds": seeds,
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
    parser.add_argument("--post-rent-binding", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluator = _load_evaluator()

    def gpu_query(field: str) -> str:
        import subprocess

        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            text=True,
        )
        values = [line.strip() for line in output.splitlines() if line.strip()]
        if len(values) != 1:
            raise ValueError(f"capacity canary requires exactly one live GPU {field}")
        return values[0]

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
        return sum(float(line) for line in output.splitlines() if line.strip())

    def compute_process_names() -> list[str]:
        import subprocess

        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=process_name",
                "--format=csv,noheader",
            ],
            text=True,
        )
        return [line.strip() for line in output.splitlines() if line.strip()]

    result = run_canary(
        args.pool_manifest.read_bytes(),
        args.data_dir,
        args.family,
        json.loads(args.runtime_identity.read_text(encoding="utf-8")),
        args.post_rent_binding.read_bytes(),
        args.receipt.read_bytes(),
        args.journal,
        lambda payload: evaluator._post_json(
            args.base_url, args.api_key, args.timeout_seconds, payload
        ),
        lambda: gpu_query("uuid"),
        lambda: int(gpu_query("memory.total")),
        compute_process_names,
        memory_used_mib,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

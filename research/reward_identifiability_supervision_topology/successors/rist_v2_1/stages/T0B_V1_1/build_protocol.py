"""Build the CPU-frozen RIST-v2.1 T0b v1.1 tasks and non-execution manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL = "RIST-T0B-RUNTIME-TOKENS-v1.1"
CALIBRATION_SEED = 2026080311
REQUEST_SEED = 11301
RUNTIME_SOURCE_COMMIT = "b6d7bdcfb0ee7be89ec1c6e16433e33ba5546db7"
SERVE_SOURCE_SHA256 = "ad84b6af12ba0fef73c7b6374031fd447dc44c140b9f59de456063684c81bf00"
PARENT_CLIENT_SHA256 = "effd096d4e8e0876dd70f0bd43fb615b349e7c9574fc296466fbcbe99b36e9b6"
CLIENT_WRAPPER_SHA256 = "a2eb9602a63ff86f5747eb21607ea9876e788ff633faab68e804c3b23de842fe"
MODEL_LOCK_SHA256 = "ff98d15528a9719e538ed47fc3be5c3413f0c8610f0bbc5bd79120dc71cb7f2a"
PRIOR_TASKS_SHA256 = "683fe8a664b29a80a44fe89809d0f1833e144f1a382e8bc98302612094b83931"
TOOL_NAMES = ("scan_registry", "inspect_candidate", "verify_route", "submit_route")
MODEL_REVISIONS = {
    "qwen3_0_6b": "c1899de289a04d12100db370d81485cdf75e47ca",
    "gemma4_e2b_it": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
}


def build_tasks() -> dict[str, Any]:
    tasks = []
    for task_index in range(8):
        turns = []
        for turn_index, tool_name in enumerate(TOOL_NAMES):
            code = hashlib.sha256(
                f"v1.1:{CALIBRATION_SEED}:{task_index}:{turn_index}".encode()
            ).hexdigest()[:12]
            turns.append(
                {
                    "turn_index": turn_index,
                    "expected_tool": tool_name,
                    "expected_code": code,
                }
            )
        tasks.append(
            {
                "id": f"t0b-v1-1-calibration-{task_index:02d}",
                "sample_index": task_index,
                "nonce": hashlib.sha256(
                    f"v1.1-nonce:{CALIBRATION_SEED}:{task_index}".encode()
                ).hexdigest()[:16],
                "turns": turns,
            }
        )
    return {"protocol": PROTOCOL, "seed": CALIBRATION_SEED, "tasks": tasks}


def write_protocol(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = output_dir / "calibration_tasks.json"
    tasks_path.write_text(
        json.dumps(build_tasks(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "FROZEN_CPU_READY_AWAITING_SEPARATE_GPU_AUTHORIZATION",
        "runtime_source_commit": RUNTIME_SOURCE_COMMIT,
        "serve_source_sha256": SERVE_SOURCE_SHA256,
        "parent_client_sha256": PARENT_CLIENT_SHA256,
        "client_wrapper_sha256": CLIENT_WRAPPER_SHA256,
        "model_acquisition_lock_sha256": MODEL_LOCK_SHA256,
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "prior_tasks_sha256": PRIOR_TASKS_SHA256,
        "prior_protocol_inputs_permitted": False,
        "task_count": 8,
        "turns_per_task": 4,
        "expected_runtime_rows_per_model": 32,
        "cases_per_turn": 8,
        "model_order": ["qwen3_0_6b", "gemma4_e2b_it"],
        "model_revisions": MODEL_REVISIONS,
        "serve": {
            "tp_size": 1,
            "world_size": 1,
            "attn_backend": "native",
            "eager_decode": True,
            "disable_thinking": True,
            "max_running_prompts": 1,
            "port": 8000,
        },
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 96},
        "request_seed": REQUEST_SEED,
        "retry_limit": 0,
        "maximum_requested_gpu_seconds": 1800,
        "training_permitted": False,
        "heldout_data_permitted": False,
        "bfcl_content_permitted": False,
        "checkpoint_download_permitted": False,
        "tokenizer_hash_match_required_before_serving": True,
        "actual_response_tokens_required": True,
        "execution_authorized": False,
    }
    (output_dir / "EXECUTION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_protocol(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

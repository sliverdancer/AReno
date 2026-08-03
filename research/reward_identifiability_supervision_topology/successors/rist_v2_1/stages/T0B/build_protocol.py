"""Build the frozen RIST-v2.1 T0b calibration tasks and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL = "RIST-T0B-RUNTIME-TOKENS-v1.0"
CALIBRATION_SEED = 20260803
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
                f"{CALIBRATION_SEED}:{task_index}:{turn_index}".encode()
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
                "id": f"t0b-calibration-{task_index:02d}",
                "sample_index": task_index,
                "nonce": hashlib.sha256(
                    f"nonce:{CALIBRATION_SEED}:{task_index}".encode()
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
        "status": "FROZEN_AUTHORIZED_AWAITING_REACHABLE_REMOTE",
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
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
        "request_seed": 9301,
        "retry_limit": 0,
        "gpu_time_limit_seconds": 1800,
        "training_permitted": False,
        "heldout_data_permitted": False,
        "bfcl_content_permitted": False,
        "checkpoint_download_permitted": False,
        "tokenizer_hash_match_required_before_serving": True,
        "execution_authorized": True,
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

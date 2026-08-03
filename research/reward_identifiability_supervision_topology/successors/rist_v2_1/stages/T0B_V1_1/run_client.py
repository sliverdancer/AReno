"""Collect T0b v1.1 runtime-token rows through the frozen zero-retry client."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

PROTOCOL = "RIST-T0B-RUNTIME-TOKENS-v1.1"


def _load_parent_client():
    path = Path(__file__).parents[1] / "T0B" / "run_client.py"
    spec = importlib.util.spec_from_file_location("rist_t0b_v1_0_parent_client", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen T0b parent client")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROTOCOL = PROTOCOL
    return module


def collect(
    *,
    base_url: str,
    api_key: str,
    model_cell: str,
    tasks_path: Path,
    manifest_path: Path,
    journal_path: Path,
    result_path: Path,
    timeout_seconds: float,
    post_json: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return _load_parent_client().collect(
        base_url=base_url,
        api_key=api_key,
        model_cell=model_cell,
        tasks_path=tasks_path,
        manifest_path=manifest_path,
        journal_path=journal_path,
        result_path=result_path,
        timeout_seconds=timeout_seconds,
        post_json=post_json,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-cell", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    result = collect(
        base_url=args.base_url,
        api_key=args.api_key,
        model_cell=args.model_cell,
        tasks_path=args.tasks,
        manifest_path=args.manifest,
        journal_path=args.journal,
        result_path=args.result,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure"] is None and result["runtime_row_count"] == 32 else 2


if __name__ == "__main__":
    raise SystemExit(main())

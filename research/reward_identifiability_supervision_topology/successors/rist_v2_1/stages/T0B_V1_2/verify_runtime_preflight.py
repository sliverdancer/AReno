"""Fail closed unless the frozen runtime and compiled extension are importable."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(
    repo_root: Path,
    stage_dir: Path,
    expected_extension_sha256: str,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    stage_dir = stage_dir.resolve()
    manifest = json.loads((stage_dir / "EXECUTION_MANIFEST.json").read_text())
    paths = {
        "serve_source": repo_root / "areno" / "cli" / "serve.py",
        "agentic_source": repo_root / "areno" / "api" / "agentic.py",
        "capture_source": stage_dir.parent / "T0" / "capture_mask_fixture.py",
        "evaluator_source": stage_dir.parent / "T0" / "evaluate_mask_fixture.py",
        "parent_client": stage_dir.parent / "T0B" / "run_client.py",
        "client_wrapper": stage_dir / "run_client.py",
        "model_lock": stage_dir.parent / "T0B" / "MODEL_ACQUISITION_LOCK.json",
        "tasks": stage_dir / "calibration_tasks.json",
    }
    checks = {
        "execution_closed": manifest.get("execution_authorized") is False,
        "training_closed": manifest.get("training_permitted") is False,
        "heldout_closed": manifest.get("heldout_data_permitted") is False,
        "bfcl_closed": manifest.get("bfcl_content_permitted") is False,
        "downloads_closed": manifest.get("checkpoint_download_permitted") is False,
        "zero_retry": manifest.get("retry_limit") == 0,
        "serve_source_exact": _sha256(paths["serve_source"])
        == manifest.get("serve_source_sha256"),
        "agentic_source_exact": _sha256(paths["agentic_source"])
        == manifest.get("agentic_source_sha256"),
        "capture_source_exact": _sha256(paths["capture_source"])
        == manifest.get("capture_source_sha256"),
        "evaluator_source_exact": _sha256(paths["evaluator_source"])
        == manifest.get("evaluator_source_sha256"),
        "parent_client_exact": _sha256(paths["parent_client"])
        == manifest.get("parent_client_sha256"),
        "client_wrapper_exact": _sha256(paths["client_wrapper"])
        == manifest.get("client_wrapper_sha256"),
        "model_lock_exact": _sha256(paths["model_lock"])
        == manifest.get("model_acquisition_lock_sha256"),
        "tasks_exact": _sha256(paths["tasks"]) == manifest.get("tasks_sha256"),
    }
    extension_path = None
    extension_error = None
    try:
        extension = importlib.import_module("areno.accel._extension").extension()
        extension_path = Path(extension.__file__).resolve()
        checks["compiled_extension_importable"] = extension_path.is_file()
        checks["compiled_extension_hash_exact"] = (
            _sha256(extension_path) == expected_extension_sha256
        )
    except Exception as exc:
        checks["compiled_extension_importable"] = False
        checks["compiled_extension_hash_exact"] = False
        extension_error = f"{type(exc).__name__}: {exc}"
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": 1,
        "protocol": "RIST-T0B-V1.2-RUNTIME-PREFLIGHT",
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "extension_path": str(extension_path) if extension_path else None,
        "extension_sha256": _sha256(extension_path) if extension_path else None,
        "extension_error": extension_error,
        "model_accessed": False,
        "inference_run": False,
        "training_run": False,
        "gpu_used": False,
        "heldout_opened": False,
        "bfcl_content_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--expected-extension-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.repo_root, args.stage_dir, args.expected_extension_sha256)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

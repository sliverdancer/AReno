"""Fail-closed CPU verifier for a frozen T0b v1.1 deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_file(repo_root: Path, commit: str, path: Path) -> bytes:
    relative = path.relative_to(repo_root).as_posix()
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout


def verify(repo_root: Path, stage_dir: Path, archive_path: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    stage_dir = stage_dir.resolve()
    manifest = _load(stage_dir / "EXECUTION_MANIFEST.json")
    bundle = _load(stage_dir / "DEPLOYMENT_BUNDLE_MANIFEST.json")
    tasks_path = stage_dir / "calibration_tasks.json"
    tasks = _load(tasks_path)
    prior_tasks = _load(stage_dir.parent / "T0B" / "calibration_tasks.json")
    serve_path = repo_root / "areno" / "cli" / "serve.py"
    parent_client = stage_dir.parent / "T0B" / "run_client.py"
    client_wrapper = stage_dir / "run_client.py"
    model_lock_path = stage_dir.parent / "T0B" / "MODEL_ACQUISITION_LOCK.json"
    model_lock = _load(model_lock_path)
    checks: dict[str, bool] = {
        "protocol_exact": manifest.get("protocol") == "RIST-T0B-RUNTIME-TOKENS-v1.1",
        "execution_closed": manifest.get("execution_authorized") is False
        and bundle.get("execution_authorized") is False,
        "training_closed": manifest.get("training_permitted") is False,
        "heldout_closed": manifest.get("heldout_data_permitted") is False,
        "bfcl_closed": manifest.get("bfcl_content_permitted") is False,
        "downloads_closed": manifest.get("checkpoint_download_permitted") is False,
        "zero_retry": int(manifest.get("retry_limit", -1)) == 0,
        "task_hash_exact": _sha256(tasks_path) == manifest.get("tasks_sha256"),
        "serve_hash_exact": _sha256(serve_path) == manifest.get("serve_source_sha256"),
        "parent_client_hash_exact": _sha256(parent_client) == manifest.get("parent_client_sha256"),
        "client_wrapper_hash_exact": _sha256(client_wrapper) == manifest.get("client_wrapper_sha256"),
        "model_lock_hash_exact": _sha256(model_lock_path)
        == manifest.get("model_acquisition_lock_sha256"),
        "model_revisions_exact": manifest.get("model_revisions")
        == {
            cell: row["revision"]
            for cell, row in model_lock.get("models", {}).items()
        },
        "runtime_commit_serve_exact": hashlib.sha256(
            _git_file(repo_root, str(manifest["runtime_source_commit"]), serve_path)
        ).hexdigest()
        == manifest.get("serve_source_sha256"),
        "deployment_commit_manifest_exact": _git_file(
            repo_root,
            str(bundle["deployment_commit"]),
            stage_dir / "EXECUTION_MANIFEST.json",
        )
        == (stage_dir / "EXECUTION_MANIFEST.json").read_bytes(),
    }
    current_rows = list(tasks.get("tasks", []))
    prior_rows = list(prior_tasks.get("tasks", []))
    current_nonces = {str(row.get("nonce")) for row in current_rows}
    prior_nonces = {str(row.get("nonce")) for row in prior_rows}
    current_codes = {
        str(turn.get("expected_code")) for row in current_rows for turn in row.get("turns", [])
    }
    prior_codes = {
        str(turn.get("expected_code")) for row in prior_rows for turn in row.get("turns", [])
    }
    checks.update(
        {
            "eight_fresh_tasks": len(current_rows) == 8 and len(current_nonces) == 8,
            "four_turns_each": all(len(row.get("turns", [])) == 4 for row in current_rows),
            "fresh_nonces": current_nonces.isdisjoint(prior_nonces),
            "fresh_codes": current_codes.isdisjoint(prior_codes) and len(current_codes) == 32,
            "v1_0_inputs_forbidden": manifest.get("prior_protocol_inputs_permitted") is False,
            "actual_tokens_required": manifest.get("actual_response_tokens_required") is True,
        }
    )
    archive_result: dict[str, Any] | None = None
    if archive_path is not None:
        archive_path = archive_path.resolve()
        with tarfile.open(archive_path, mode="r:") as handle:
            members = [member.name for member in handle.getmembers()]
        forbidden_suffixes = tuple(str(value) for value in bundle["forbidden_suffixes"])
        archive_prefix = str(bundle["archive_prefix"])
        archive_root = archive_prefix.rstrip("/")
        archive_result = {
            "path": str(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
            "member_count": len(members),
        }
        checks.update(
            {
                "archive_size_exact": archive_result["size_bytes"]
                == bundle.get("archive_size_bytes"),
                "archive_hash_exact": archive_result["sha256"] == bundle.get("archive_sha256"),
                "archive_prefix_exact": all(
                    member == archive_root or member.startswith(archive_prefix)
                    for member in members
                ),
                "archive_has_no_git_metadata": all("/.git/" not in member for member in members),
                "archive_has_no_model_weights": all(
                    not member.lower().endswith(forbidden_suffixes) for member in members
                ),
            }
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": 1,
        "protocol": "RIST-T0B-V1.1-DEPLOYMENT-PREFLIGHT",
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "archive": archive_result,
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
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.repo_root, args.stage_dir, args.archive)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Capture the corrected frozen RIST-E0-v1.2 preflight."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path("/root/autodl-tmp/rist_e0_v1_2_511ed73")
PYTHON = ROOT / "venv/bin/python"
ARENO = ROOT / "venv/bin/areno"
EVIDENCE = ROOT / "evidence"
STAGE = ROOT / (
    "research/reward_identifiability_supervision_topology/successors/"
    "rist_v1_1/stages/E0_v1_2"
)
DEPLOYED_HEAD = "511ed73c2298fc26245dbde9fcf42021a08ffba4"
REQUIRED_SOURCE = "626fd7ba96d13f56a76082b4d675ce90205494eb"
EXPECTED_TASKS_SHA = "c0a4f8d8fa4c15765b57702797f44fcf2bbc0e637b570141d439ad465fce6c72"


def run(name: str, command: list[str]) -> dict[str, object]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    record: dict[str, object] = {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    (EVIDENCE / f"preflight_{name}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    extension = run(
        "registered_extension",
        [
            str(PYTHON),
            "-c",
            (
                "from pathlib import Path; "
                "import areno.accel._areno_accel as ext; import areno; "
                f"root=Path({str(ROOT)!r}).resolve(); "
                "ext_path=Path(ext.__file__).resolve(); "
                "areno_path=Path(areno.__file__).resolve(); "
                "ext_path.relative_to(root); areno_path.relative_to(root); "
                "print(ext_path); print(areno_path)"
            ),
        ],
    )
    cuda = run(
        "cuda",
        [
            str(PYTHON),
            "-c",
            (
                "import torch; assert torch.cuda.is_available(); "
                "print(torch.__version__); print(torch.version.cuda); "
                "print(torch.cuda.get_device_name(0))"
            ),
        ],
    )
    check = run("areno_check", [str(ARENO), "check"])
    head = run("source_head", ["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    tracked = run(
        "tracked_clean",
        ["git", "-C", str(ROOT), "diff", "--quiet", "--exit-code"],
    )
    staged = run(
        "index_clean",
        ["git", "-C", str(ROOT), "diff", "--cached", "--quiet", "--exit-code"],
    )
    execution_diff = run(
        "execution_diff",
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--quiet",
            REQUIRED_SOURCE + "..HEAD",
            "--",
            str(STAGE.relative_to(ROOT) / "run_e0_canary.py"),
            str(STAGE.relative_to(ROOT) / "validate_e0.py"),
            str(STAGE.relative_to(ROOT) / "canary_tasks.json"),
        ],
    )
    gpu_processes = run(
        "gpu_processes",
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
    )
    tasks_sha = sha256(STAGE / "canary_tasks.json")
    model_verify_complete = EVIDENCE / "model_verify.complete"
    passed = all(
        [
            extension["returncode"] == 0,
            cuda["returncode"] == 0,
            check["returncode"] == 0,
            head["returncode"] == 0 and str(head["stdout"]).strip() == DEPLOYED_HEAD,
            tracked["returncode"] == 0,
            staged["returncode"] == 0,
            execution_diff["returncode"] == 0,
            gpu_processes["returncode"] == 0
            and not str(gpu_processes["stdout"]).strip(),
            tasks_sha == EXPECTED_TASKS_SHA,
            model_verify_complete.is_file(),
        ]
    )
    payload = {
        "schema_version": 1,
        "protocol": "RIST-E0-v1.2",
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "passed": passed,
        "registered_extension_passed": extension["returncode"] == 0,
        "cuda_passed": cuda["returncode"] == 0,
        "areno_check_passed": check["returncode"] == 0,
        "deployed_head_passed": head["returncode"] == 0
        and str(head["stdout"]).strip() == DEPLOYED_HEAD,
        "tracked_source_clean": tracked["returncode"] == 0,
        "index_clean": staged["returncode"] == 0,
        "execution_bytes_match_required_source": execution_diff["returncode"] == 0,
        "gpu_process_list_clean": gpu_processes["returncode"] == 0
        and not str(gpu_processes["stdout"]).strip(),
        "model_manifests_verified": model_verify_complete.is_file(),
        "canary_tasks_sha256": tasks_sha,
        "deployed_head": DEPLOYED_HEAD,
        "required_source_commit": REQUIRED_SOURCE,
        "serving_started": False,
        "training_performed": False,
    }
    (EVIDENCE / "preflight_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

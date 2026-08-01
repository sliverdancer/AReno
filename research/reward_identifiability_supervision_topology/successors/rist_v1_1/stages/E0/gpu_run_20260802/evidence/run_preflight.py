"""Capture the frozen RIST-E0-v1.1 remote preflight without starting serving."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path


ROOT = Path("/root/autodl-tmp/rist_e0_05da707")
PYTHON = ROOT / "venv/bin/python"
ARENO = ROOT / "venv/bin/areno"
SOURCE = ROOT / "source"
EVIDENCE = ROOT / "evidence"
EXPECTED_COMMIT = "05da707f19df590f613cd2403746495460809253"


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


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    frozen_import = run(
        "frozen_import",
        [str(PYTHON), "-c", "import areno_accel"],
    )
    actual_import = run(
        "actual_import",
        [
            str(PYTHON),
            "-c",
            (
                "import areno.accel._areno_accel as ext; "
                "import areno; "
                "print(ext.__file__); print(areno.__file__)"
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
    head = run("source_head", ["git", "-C", str(SOURCE), "rev-parse", "HEAD"])
    clean = run("source_clean", ["git", "-C", str(SOURCE), "status", "--porcelain"])
    gpu_processes = run(
        "gpu_processes",
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
    )
    head_ok = head["returncode"] == 0 and str(head["stdout"]).strip() == EXPECTED_COMMIT
    clean_ok = clean["returncode"] == 0 and not str(clean["stdout"]).strip()
    gpu_clean = gpu_processes["returncode"] == 0 and not str(gpu_processes["stdout"]).strip()
    passed = all(
        [
            frozen_import["returncode"] == 0,
            actual_import["returncode"] == 0,
            cuda["returncode"] == 0,
            check["returncode"] == 0,
            head_ok,
            clean_ok,
            gpu_clean,
        ]
    )
    payload = {
        "schema_version": 1,
        "protocol": "RIST-E0-v1.1",
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "passed": passed,
        "frozen_import_passed": frozen_import["returncode"] == 0,
        "actual_extension_import_passed": actual_import["returncode"] == 0,
        "cuda_passed": cuda["returncode"] == 0,
        "areno_check_passed": check["returncode"] == 0,
        "source_head_passed": head_ok,
        "source_clean_passed": clean_ok,
        "gpu_process_list_clean": gpu_clean,
        "expected_commit": EXPECTED_COMMIT,
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

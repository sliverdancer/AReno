"""Capture RIST-P2.1-v1.0 preflight without parsing qualification rows."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path("/root/autodl-tmp/rist_p2_1_0f1024b")
PYTHON = ROOT / "venv/bin/python"
ARENO = ROOT / "venv/bin/areno"
EVIDENCE = ROOT / "evidence"
STAGE = ROOT / (
    "research/reward_identifiability_supervision_topology/successors/"
    "rist_v1_1/stages/P2_1"
)
DATA = ROOT / "research/reward_identifiability_supervision_topology/stages/P1/data"
E0 = ROOT / (
    "research/reward_identifiability_supervision_topology/successors/rist_v1_1/"
    "stages/E0_v1_2/gpu_run_20260802/evidence"
)
DEPLOYED_HEAD = "0f1024bfcb3deb18a87a8eefa042f17d4a56e43a"
REQUIRED_SOURCE = "a9817fad60ca5280913aa23622fbe6f51d7b8d41"
QUALIFICATION_SHA = "8018137606e12da0f0096ac86f11312d94d631326965198783ebf9cecc94570f"
E0_STAGE_SHA = "861c0e8955550d7bb832ccaee1ed5e8050ef514ecc052773b81520a966cd01ed"
E0_AUDIT_SHA = "5977a7cc3e89d27f0da157e069c40f4c7ca76a7e9e587b44a3a8a3c51d36f24f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    extension = run(
        "registered_extension",
        [
            str(PYTHON),
            "-c",
            (
                "from pathlib import Path; import areno.accel._areno_accel as ext; "
                "import areno; "
                f"root=Path({str(ROOT)!r}).resolve(); "
                "Path(ext.__file__).resolve().relative_to(root); "
                "Path(areno.__file__).resolve().relative_to(root); "
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
    head = run("source_head", ["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    tracked = run("tracked_clean", ["git", "-C", str(ROOT), "diff", "--quiet"])
    staged = run("index_clean", ["git", "-C", str(ROOT), "diff", "--cached", "--quiet"])
    execution = run(
        "execution_diff",
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--quiet",
            REQUIRED_SOURCE + "..HEAD",
            "--",
            str(STAGE.relative_to(ROOT) / "run_p2_1_client.py"),
            str(STAGE.relative_to(ROOT) / "analyze_p2_1.py"),
        ],
    )
    gpu = run(
        "gpu_processes",
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
    )
    e0_stage_hash = sha256(E0 / "stage_result.json")
    e0_audit_hash = sha256(E0 / "audit_result.json")
    e0_stage = json.loads((E0 / "stage_result.json").read_text(encoding="utf-8"))
    e0_audit = json.loads((E0 / "audit_result.json").read_text(encoding="utf-8"))
    qualification_hash = sha256(DATA / "qualification.jsonl")
    model_verify = (EVIDENCE / "model_verify.complete").is_file()
    passed = all(
        [
            extension["returncode"] == 0,
            cuda["returncode"] == 0,
            check["returncode"] == 0,
            head["returncode"] == 0 and str(head["stdout"]).strip() == DEPLOYED_HEAD,
            tracked["returncode"] == 0,
            staged["returncode"] == 0,
            execution["returncode"] == 0,
            gpu["returncode"] == 0 and not str(gpu["stdout"]).strip(),
            e0_stage_hash == E0_STAGE_SHA,
            e0_audit_hash == E0_AUDIT_SHA,
            e0_stage.get("decision")
            == "PASS_E0_INFRASTRUCTURE_TO_P2_1_PROTOCOL_FREEZE",
            e0_audit.get("passed") is True,
            qualification_hash == QUALIFICATION_SHA,
            model_verify,
        ]
    )
    payload = {
        "schema_version": 1,
        "protocol": "RIST-P2.1-v1.0",
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "passed": passed,
        "registered_extension_passed": extension["returncode"] == 0,
        "cuda_passed": cuda["returncode"] == 0,
        "areno_check_passed": check["returncode"] == 0,
        "deployed_head": DEPLOYED_HEAD,
        "required_source_commit": REQUIRED_SOURCE,
        "deployed_head_passed": head["returncode"] == 0
        and str(head["stdout"]).strip() == DEPLOYED_HEAD,
        "tracked_source_clean": tracked["returncode"] == 0,
        "index_clean": staged["returncode"] == 0,
        "execution_bytes_match_required_source": execution["returncode"] == 0,
        "gpu_process_list_clean": gpu["returncode"] == 0
        and not str(gpu["stdout"]).strip(),
        "e0_stage_result_sha256": e0_stage_hash,
        "e0_audit_result_sha256": e0_audit_hash,
        "e0_prerequisite_passed": e0_stage.get("stage_status") == "PASS"
        and e0_audit.get("passed") is True,
        "qualification_sha256": qualification_hash,
        "qualification_rows_parsed": False,
        "heldout_accessed": False,
        "model_manifests_verified": model_verify,
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

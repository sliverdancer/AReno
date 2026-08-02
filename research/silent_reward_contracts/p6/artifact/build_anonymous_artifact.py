"""Build a deterministic, allowlisted ARCA anonymous artifact archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from research.silent_reward_contracts.p6.artifact.verify_artifact import (
    MANIFEST_NAME,
    scan_text,
    sha256_bytes,
    verify_directory,
)


ARCHIVE_ROOT = "ARCA-anonymous-artifact-v0.1"
PAYLOAD_PATHS = (
    "LICENSE",
    "research/silent_reward_contracts/__init__.py",
    "research/silent_reward_contracts/arca.py",
    "research/silent_reward_contracts/evaluate_auditor.py",
    "research/silent_reward_contracts/summarize_p3.py",
    "research/silent_reward_contracts/p0/direct_substitute_matrix.csv",
    "research/silent_reward_contracts/p0/source_manifest.json",
    "research/silent_reward_contracts/p3/FREEZE.md",
    "research/silent_reward_contracts/p3/dev_cases.json",
    "research/silent_reward_contracts/p3/heldout_cases.json",
    "research/silent_reward_contracts/p3/upstream_manifest.json",
    "research/silent_reward_contracts/p3/artifacts/dev/cases.csv",
    "research/silent_reward_contracts/p3/artifacts/dev/metrics.json",
    "research/silent_reward_contracts/p3/artifacts/heldout/cases.csv",
    "research/silent_reward_contracts/p3/artifacts/heldout/metrics.json",
    "research/silent_reward_contracts/p3/artifacts/summary/gate_summary.json",
    "research/silent_reward_contracts/p3/artifacts/replication_areal.json",
    "research/silent_reward_contracts/p4/GPU_GATE_DECISION_20260801.md",
    "research/silent_reward_contracts/p4/gpu_gate_decision.json",
    "research/silent_reward_contracts/p5/FREEZE.md",
    "research/silent_reward_contracts/p5/RESULTS.md",
    "research/silent_reward_contracts/p5/upstream_manifest.json",
    "research/silent_reward_contracts/p5/external_validation.py",
    "research/silent_reward_contracts/p5/artifacts/external_cases.csv",
    "research/silent_reward_contracts/p5/artifacts/external_cases.json",
    "research/silent_reward_contracts/p5/artifacts/external_evidence.json",
    "research/silent_reward_contracts/p5/artifacts/external_metrics.json",
    "research/silent_reward_contracts/p5/artifacts/framework_inspections.csv",
    "research/silent_reward_contracts/p6/FREEZE.md",
    "research/silent_reward_contracts/p6/LITERATURE_REFRESH.md",
    "research/silent_reward_contracts/p6/paper/main.tex",
    "research/silent_reward_contracts/p6/paper/references.bib",
    "research/silent_reward_contracts/p6/paper/style_manifest.json",
    "research/silent_reward_contracts/p6/paper_claim_audit.json",
    "research/silent_reward_contracts/p6/validate_paper_claims.py",
    "research/silent_reward_contracts/p6/artifact/__init__.py",
    "research/silent_reward_contracts/p6/artifact/build_anonymous_artifact.py",
    "research/silent_reward_contracts/p6/artifact/README_ARTIFACT.md",
    "research/silent_reward_contracts/p6/artifact/verify_artifact.py",
    "tests/test_arca_p5_cpu.py",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def configured_identities(repo_root: Path) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("user.name", "user.email"):
        result = subprocess.run(
            ["git", "config", "--get", key],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        if len(value) >= 4:
            values.append(value)
    return tuple(values)


def copy_payload(repo_root: Path, staging_root: Path) -> list[dict[str, object]]:
    identities = configured_identities(repo_root)
    entries: list[dict[str, object]] = []
    for relative_path in PAYLOAD_PATHS:
        source = repo_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(relative_path)
        data = source.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = ""
        violations = scan_text(relative_path, text) if text else []
        for identity in identities:
            if identity.casefold() in text.casefold():
                violations.append(f"{relative_path}: configured author identity")
        if violations:
            raise ValueError("; ".join(violations))
        destination = staging_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        entries.append(
            {"path": relative_path, "sha256": sha256_bytes(data), "size": len(data)}
        )
    return sorted(entries, key=lambda entry: str(entry["path"]))


def write_manifest(staging_root: Path, entries: list[dict[str, object]]) -> None:
    payload_digest = hashlib.sha256()
    for entry in entries:
        payload_digest.update(str(entry["path"]).encode("utf-8"))
        payload_digest.update(str(entry["sha256"]).encode("ascii"))
    manifest = {
        "schema_version": "arca.anonymous-artifact.v1",
        "protocol_id": "ARCA-P6-PAPER-v0.1",
        "archive_root": ARCHIVE_ROOT,
        "file_count": len(entries),
        "payload_sha256": payload_digest.hexdigest(),
        "files": entries,
    }
    (staging_root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def deterministic_tar_gz(staging_root: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted(staging_root.rglob("*")):
                    if not path.is_file():
                        continue
                    data = path.read_bytes()
                    relative = path.relative_to(staging_root).as_posix()
                    info = tarfile.TarInfo(f"{ARCHIVE_ROOT}/{relative}")
                    info.size = len(data)
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(data))


def build_archive(output_path: Path, repo_root: Path | None = None) -> dict[str, object]:
    repo_root = (repo_root or repository_root()).resolve()
    with tempfile.TemporaryDirectory(prefix="arca-p6-build-") as temporary:
        staging_root = Path(temporary) / ARCHIVE_ROOT
        staging_root.mkdir()
        entries = copy_payload(repo_root, staging_root)
        write_manifest(staging_root, entries)
        audit = verify_directory(staging_root)
        deterministic_tar_gz(staging_root, output_path)
    archive_data = output_path.read_bytes()
    return {
        **audit,
        "archive": output_path.name,
        "archive_sha256": sha256_bytes(archive_data),
        "archive_size": len(archive_data),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = build_archive(args.output)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
from pathlib import Path
import tarfile

import pytest

from research.silent_reward_contracts.p6.artifact.build_anonymous_artifact import (
    ARCHIVE_ROOT,
    build_archive,
)
from research.silent_reward_contracts.p6.artifact.verify_artifact import (
    scan_text,
    verify_directory,
)
from research.silent_reward_contracts.p6.validate_paper_claims import validate_claims


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p6_anonymous_artifact_is_deterministic_and_verifiable(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    first_result = build_archive(first)
    second_result = build_archive(second)

    assert first_result["status"] == "PASS_ANONYMOUS_ARTIFACT_AUDIT"
    assert first_result["payload_sha256"] == second_result["payload_sha256"]
    assert _sha256(first) == _sha256(second)

    extracted = tmp_path / "extracted"
    with tarfile.open(first, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    verified = verify_directory(extracted / ARCHIVE_ROOT)
    assert verified["status"] == "PASS_ANONYMOUS_ARTIFACT_AUDIT"


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("input=/mnt/private/project/file.json", "local Linux path"),
        ("ssh user@example.invalid", "SSH endpoint"),
        ("api_key='not-a-real-key'", "API token assignment"),
    ],
)
def test_p6_anonymity_scanner_rejects_forbidden_text(text: str, label: str) -> None:
    violations = scan_text("sample.txt", text)
    assert any(label in violation for violation in violations)


def test_p6_manuscript_numbers_reconcile_with_json(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "paper_claim_audit.json"
    result = validate_claims(repo_root, output)
    assert result["status"] == "PASS_PAPER_CLAIM_AUDIT"
    assert output.is_file()

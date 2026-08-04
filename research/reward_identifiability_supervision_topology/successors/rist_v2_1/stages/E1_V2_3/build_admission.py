"""Build the structural E1 v2.3 admission only from C0 v2.3 transport PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
MODELS = {"qwen3": "Qwen/Qwen3-0.6B", "gemma4": "google/gemma-4-E2B-it"}
REVISIONS = {
    "qwen3": "c1899de289a04d12100db370d81485cdf75e47ca",
    "gemma4": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
}
TOKENIZER_SNAPSHOTS = {
    "qwen3": "c83c7f983e1204841852a4cb47cff31dfd829437c80dccc55dd52d0c8fe532b1",
    "gemma4": "7c813a44e67aa09d81001db777c261d858417b45ce0204bc6a32f0b7b96720f7",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _clean_head() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("E1 admission requires a clean frozen worktree")
    return head


def _validate_transport(value: dict[str, Any]) -> dict[str, str]:
    if not (
        value.get("protocol") == "RIST-C0-v2.3-QUALIFICATION-TRANSPORT-v1"
        and value.get("passed") is True
        and value.get("decision") == "PASS_C0_V2_3_TO_E1_CAPACITY"
        and value.get("whole_cell_selection_only") is True
        and value.get("heldout_accessed") is False
        and value.get("bfcl_accessed") is False
        and value.get("training_performed") is False
        and isinstance(value.get("gpu_uuid"), str)
        and value["gpu_uuid"].startswith("GPU-")
    ):
        raise PermissionError("E1 requires the exact C0 v2.3 qualification transport PASS")
    mapping = value.get("common_resolution_map", {})
    if not isinstance(mapping, dict) or set(mapping.values()) != {"low", "high"}:
        raise PermissionError("C0 transport map must contain low and high bands")
    counts = Counter(mapping.values())
    declared = value.get("band_cell_counts", {})
    if counts["low"] < 2 or counts["high"] < 2 or declared != dict(counts):
        raise PermissionError("E1 requires two transported whole cells per band")
    if len(value.get("families", [])) != 2 or {
        row.get("family") for row in value["families"]
    } != {"qwen3", "gemma4"} or not all(row.get("passed") is True for row in value["families"]):
        raise PermissionError("E1 requires both family transport gates")
    return {str(cell): str(band) for cell, band in mapping.items()}


def build_admission(
    *,
    transport_result_path: Path,
    d3_manifest_path: Path,
    d3_train_path: Path,
    output_root: Path,
    source_commit: str | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("E1 v2.3 admission root must be fresh")
    transport = json.loads(transport_result_path.read_text(encoding="utf-8"))
    resolution_map = _validate_transport(transport)
    d3_manifest = json.loads(d3_manifest_path.read_text(encoding="utf-8"))
    if not (
        d3_manifest.get("protocol") == "RIST-D3-v2.1"
        and d3_manifest.get("split") == "train"
        and d3_manifest.get("count") == 32
        and d3_manifest.get("heldout_data_opened") is False
        and d3_manifest.get("training_performed") is False
        and d3_manifest.get("sha256") == _sha256(d3_train_path)
    ):
        raise ValueError("unexpected or modified D3 train source")
    rows = [json.loads(line) for line in d3_train_path.read_text(encoding="utf-8").splitlines() if line]
    all_counts = Counter(str(row.get("structural_cell")) for row in rows)
    if len(rows) != 32 or len(all_counts) != 8 or set(all_counts.values()) != {4}:
        raise ValueError("D3 source must contain four tasks in each of eight cells")
    filtered = []
    for source in rows:
        cell = str(source["structural_cell"])
        if cell in resolution_map:
            filtered.append({**source, "resolution_band": resolution_map[cell]})
    selected_counts = Counter(str(row["structural_cell"]) for row in filtered)
    if set(selected_counts) != set(resolution_map) or set(selected_counts.values()) != {4}:
        raise ValueError("E1 filtering must retain every task in each selected cell")
    high_cells = sorted(cell for cell, band in resolution_map.items() if band == "high")
    selected_cell = high_cells[0]
    capacity = [row for row in filtered if row["structural_cell"] == selected_cell]
    if len(capacity) != 4 or any(row["resolution_band"] != "high" for row in capacity):
        raise ValueError("E1 capacity data must be one complete transported high cell")
    commit = source_commit if source_commit is not None else _clean_head()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("E1 source commit must be a full lowercase commit")

    filtered_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in filtered).encode()
    capacity_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in capacity).encode()
    filtered_path = output_root / "data/filtered_train.jsonl"
    capacity_path = output_root / "data/capacity_train.jsonl"
    resolution_path = output_root / "data/c0_transport_result.json"
    capacity_manifest_path = output_root / "data/capacity_train.manifest.json"
    _write_exclusive(filtered_path, filtered_bytes)
    _write_exclusive(capacity_path, capacity_bytes)
    resolution_bytes = (json.dumps(transport, sort_keys=True) + "\n").encode()
    _write_exclusive(resolution_path, resolution_bytes)
    capacity_manifest = {
        "protocol": "RIST-E1-CAPACITY-DATA-v2.1",
        "selection_rule": "LEXICOGRAPHIC_FIRST_TRANSPORTED_HIGH_CELL",
        "selection_uses_individual_outcomes": False,
        "selected_cell": selected_cell,
        "task_count": len(capacity),
        "source_train_sha256": hashlib.sha256(filtered_bytes).hexdigest(),
        "resolution_result_sha256": hashlib.sha256(resolution_bytes).hexdigest(),
        "capacity_train_sha256": hashlib.sha256(capacity_bytes).hexdigest(),
    }
    capacity_manifest_bytes = (
        json.dumps(capacity_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    _write_exclusive(capacity_manifest_path, capacity_manifest_bytes)
    admission = {
        "protocol": "RIST-E1-v2.3-C0-ADMISSION-v1",
        "passed": True,
        "decision": "PASS_C0_V2_3_GATE_TO_DEPLOYMENT_BOUND_E1",
        "source_commit": commit,
        "transport_result_sha256": _sha256(transport_result_path),
        "d3_manifest_sha256": _sha256(d3_manifest_path),
        "d3_train_sha256": _sha256(d3_train_path),
        "common_resolution_map": resolution_map,
        "band_cell_counts": dict(Counter(resolution_map.values())),
        "filtered_train_path": str(filtered_path),
        "filtered_train_sha256": hashlib.sha256(filtered_bytes).hexdigest(),
        "filtered_task_count": len(filtered),
        "capacity_train_path": str(capacity_path),
        "capacity_train_sha256": hashlib.sha256(capacity_bytes).hexdigest(),
        "capacity_task_count": len(capacity),
        "resolution_result_path": str(resolution_path),
        "resolution_result_sha256": hashlib.sha256(resolution_bytes).hexdigest(),
        "capacity_data_manifest_path": str(capacity_manifest_path),
        "capacity_data_manifest_sha256": hashlib.sha256(
            capacity_manifest_bytes
        ).hexdigest(),
        "capacity_selection_rule": "LEXICOGRAPHIC_FIRST_TRANSPORTED_HIGH_CELL",
        "selected_capacity_cell": selected_cell,
        "selection_uses_individual_outcomes": False,
        "models": MODELS,
        "model_revisions": REVISIONS,
        "tokenizer_snapshot_sha256": TOKENIZER_SNAPSHOTS,
        "gpu_uuid": transport["gpu_uuid"],
        "execution_authorized": False,
        "training_permitted": False,
        "heldout_permitted": False,
        "bfcl_permitted": False,
    }
    _write_exclusive(
        output_root / "E1_ADMISSION.json",
        (json.dumps(admission, indent=2, sort_keys=True) + "\n").encode(),
    )
    return admission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport-result", type=Path, required=True)
    parser.add_argument("--d3-manifest", type=Path, required=True)
    parser.add_argument("--d3-train", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_admission(
        transport_result_path=args.transport_result,
        d3_manifest_path=args.d3_manifest,
        d3_train_path=args.d3_train,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

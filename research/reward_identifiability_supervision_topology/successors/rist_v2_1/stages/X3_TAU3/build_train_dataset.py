"""Build outcome-blind Tau3 training records from the frozen X2 partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_rows(partitions: dict) -> list[dict]:
    if partitions.get("protocol") != "RIST-X2-TAU3-PARTITIONS-v1":
        raise ValueError("unexpected Tau3 partition protocol")
    if partitions["partitions"].get("confirmatory") != []:
        raise ValueError("Tau3 confirmatory partition must remain empty")
    development = set(partitions["partitions"]["development"])
    training = list(partitions["partitions"]["training"])
    if development & set(training) or len(training) != len(set(training)):
        raise ValueError("Tau3 train IDs must be unique and development-disjoint")
    rows = []
    blocked_retail = []
    for namespaced_id in sorted(training):
        domain, task_id = namespaced_id.split(":", 1)
        if domain not in {"airline", "retail"} or not task_id:
            raise ValueError(f"invalid Tau3 training ID: {namespaced_id}")
        if domain == "retail":
            blocked_retail.append(namespaced_id)
            continue
        rows.append(
            {
                "id": namespaced_id,
                "split": "tau3_train",
                "domain": domain,
                "task_id": task_id,
                "prompt": (
                    "Interact with the user and the stateful Tau3 environment. "
                    "Follow the domain policy, use tools when needed, and finish "
                    "only after the user's request is resolved."
                ),
                "tau3_tag": partitions["tau3_tag"],
                "tau3_commit": partitions["tau3_commit"],
            }
        )
    if len(rows) != 22 or len(blocked_retail) != 66:
        raise ValueError("X3 requires 22 airline rows and blocks all 66 retail rows")
    return rows


def write_dataset(partition_path: Path, output: Path) -> dict:
    partition_bytes = partition_path.read_bytes()
    partitions = json.loads(partition_bytes)
    rows = build_rows(partitions)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    output.write_bytes(encoded)
    manifest = {
        "protocol": "RIST-X3-TAU3-DATA-v1",
        "partition_sha256": hashlib.sha256(partition_bytes).hexdigest(),
        "train_sha256": hashlib.sha256(encoded).hexdigest(),
        "count": len(rows),
        "blocked_retail_count": 66,
        "blocked_retail_reason": "domain-wide NL_ASSERTION confound",
        "domain_counts": {
            domain: sum(row["domain"] == domain for row in rows)
            for domain in ("airline", "retail")
        },
        "upstream_test_opened": False,
        "runtime_loader_may_parse_unified_retired_source": True,
        "task_content_copied": False,
        "execution_authorized": False,
    }
    output.with_name("manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_dataset(args.partitions, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

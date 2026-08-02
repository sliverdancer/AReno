"""Freeze Tau3 development/training IDs without using model outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DOMAINS = ("airline", "retail")
DEVELOPMENT_PER_DOMAIN = 8


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def select_partitions(
    split_ids: dict[str, list[str]],
    action_counts: dict[str, int],
    *,
    development_count: int = DEVELOPMENT_PER_DOMAIN,
) -> dict[str, Any]:
    """Split upstream train IDs by a frozen structural eligibility rule."""

    if set(split_ids) != {"train", "test", "base"}:
        raise ValueError("Tau3 split file must contain train, test, and base")
    train = [str(value) for value in split_ids["train"]]
    test = [str(value) for value in split_ids["test"]]
    if len(train) != len(set(train)) or len(test) != len(set(test)):
        raise ValueError("Tau3 split IDs must be unique")
    if set(train) & set(test):
        raise ValueError("Tau3 upstream train and test IDs overlap")
    if set(split_ids["base"]) != set(train) | set(test):
        raise ValueError("Tau3 base IDs must equal train union test")
    if not set(train) <= set(action_counts):
        raise ValueError("action metadata is missing for upstream train IDs")
    eligible = sorted(
        (task_id for task_id in train if int(action_counts[task_id]) > 0),
        key=int,
    )
    if len(eligible) < development_count:
        raise ValueError("insufficient action-bearing Tau3 development candidates")
    development = eligible[:development_count]
    training = [task_id for task_id in train if task_id not in set(development)]
    return {
        "development": development,
        "training": training,
        "confirmatory": [],
        "upstream_test_retired_count": len(test),
        "upstream_test_ids_sha256": _hash_json(sorted(test, key=int)),
        "selection_uses_model_outcomes": False,
        "selection_rule": "first_integer_sorted_train_ids_with_nonempty_reference_actions",
    }


def build_manifest(tau3_root: Path) -> dict[str, Any]:
    domain_rows = {}
    flat = {"development": [], "training": [], "confirmatory": []}
    for domain in DOMAINS:
        data_root = tau3_root / "data" / "tau2" / "domains" / domain
        split_path = data_root / "split_tasks.json"
        tasks_path = data_root / "tasks.json"
        split_ids = json.loads(split_path.read_text(encoding="utf-8"))
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        action_counts = {
            str(task["id"]): len(
                ((task.get("evaluation_criteria") or {}).get("actions") or [])
            )
            for task in tasks
        }
        selected = select_partitions(split_ids, action_counts)
        domain_rows[domain] = {
            **selected,
            "split_file_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "tasks_file_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        }
        for split in flat:
            flat[split].extend(
                f"{domain}:{task_id}" for task_id in selected[split]
            )
    if set(flat["development"]) & set(flat["training"]):
        raise ValueError("Tau3 development and training IDs overlap")
    return {
        "protocol": "RIST-X2-TAU3-PARTITIONS-v1",
        "tau3_tag": "v1.0.1",
        "tau3_commit": "fc0055dc4e0a316c3f83133267fbd6faaa770992",
        "development_per_domain": DEVELOPMENT_PER_DOMAIN,
        "domains": domain_rows,
        "partitions": flat,
        "tau3_upstream_test_retired": True,
        "confirmatory_source": "bfcl_sealed",
        "model_outcomes_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.tau3_root)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Load only the frozen RIST-v2.1 Tau3 training records."""

from __future__ import annotations

from pathlib import Path


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    path = Path(dataset_path)
    if path.name != "train.jsonl" or path.parent.name != "data":
        raise ValueError("Tau3 loader accepts only the frozen X3 data/train.jsonl")
    records = []
    for source in default_loader(dataset_path):
        record = dict(source)
        if record.get("split") != "tau3_train":
            raise ValueError("Tau3 training record must have split=tau3_train")
        if record.get("domain") != "airline":
            raise ValueError("X3 blocks retail because its reward invokes NL_ASSERTION")
        if not isinstance(record.get("task_id"), str) or not record["task_id"]:
            raise ValueError("Tau3 training record requires task_id")
        if record.get("tau3_tag") != "v1.0.1" or record.get("tau3_commit") != (
            "fc0055dc4e0a316c3f83133267fbd6faaa770992"
        ):
            raise ValueError("Tau3 source identity does not match the frozen checkout")
        records.append(record)
    if not records:
        raise ValueError("Tau3 training dataset is empty")
    return records

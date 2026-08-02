"""Normalize RIST-v2.1 train rows into prompt-bearing AReno records."""

from __future__ import annotations

from pathlib import Path


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Load only the frozen train split and construct its initial prompt."""

    path = Path(dataset_path)
    if path.name != "train.jsonl" or path.parent.name != "data":
        raise ValueError("RIST-v2.1 training loader accepts only frozen data/train.jsonl")
    records = []
    for source in default_loader(dataset_path):
        record = dict(source)
        if record.get("split") != "train":
            raise ValueError("RIST-v2.1 training record must have split=train")
        contract = record.get("prompt_contract")
        if not isinstance(contract, dict):
            raise ValueError("RIST-v2.1 record requires prompt_contract")
        visible = contract.get("initial_visible_turns")
        if not isinstance(visible, list) or not visible:
            raise ValueError("RIST-v2.1 record requires visible initial turns")
        record["prompt"] = _render_prompt(str(contract["instruction"]), visible)
        records.append(record)
    return records


def _render_prompt(instruction: str, turns: list[dict]) -> str:
    rendered = [instruction]
    for turn in turns:
        rendered.append(
            "Turn {index}: offered tools={tools}; target_label={label}; candidates={candidates}.".format(
                index=int(turn["turn_index"]) + 1,
                tools=", ".join(str(name) for name in turn["offered_tools"]),
                label=turn["target_label"],
                candidates=", ".join(
                    f"{candidate['label']}:{candidate['code']}"
                    for candidate in turn["candidate_records"]
                ),
            )
        )
    return "\n".join(rendered)

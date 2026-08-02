"""Capture canonical real-tokenizer fixtures using local checkpoint files only."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any, Callable

TOOL_NAMES = ("scan_registry", "request_hint", "verify_route", "submit_route")
CODE_VALUES = tuple(f"r{index}-{index * 7919:08x}" for index in range(8))


def _overlaps(offset: tuple[int, int], span: tuple[int, int]) -> bool:
    return offset[0] < span[1] and span[0] < offset[1]


def _inside(offset: tuple[int, int], span: tuple[int, int]) -> bool:
    return span[0] <= offset[0] and offset[1] <= span[1]


def build_fixture(
    tokenizer: Any,
    checkpoint: str,
    arg_range_fn: Callable[[Any, list[int]], tuple[int, int] | None],
) -> dict[str, Any]:
    """Build 32 canonical cases with actual tokenizer offsets and current mask."""

    vocab_payload = json.dumps(
        {
            "vocab": tokenizer.get_vocab(),
            "special_tokens_map": tokenizer.special_tokens_map,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    cases = []
    for case_index, (name, code) in enumerate(product(TOOL_NAMES, CODE_VALUES)):
        raw_call = json.dumps(
            {"name": name, "arguments": {"code": code}},
            separators=(",", ":"),
        )
        encoded = tokenizer(
            raw_call,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = [int(token_id) for token_id in encoded["input_ids"]]
        offsets = [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]]
        name_start = raw_call.index(name)
        name_span = (name_start, name_start + len(name))
        arguments_key_end = raw_call.index('"arguments":') + len('"arguments":')
        argument_span = (arguments_key_end, len(raw_call) - 1)
        roles = {
            "name_indices": [],
            "argument_indices": [],
            "other_indices": [],
            "shared_indices": [],
        }
        for token_index, offset in enumerate(offsets):
            touches_name = _overlaps(offset, name_span)
            touches_argument = _overlaps(offset, argument_span)
            if touches_name and _inside(offset, name_span):
                roles["name_indices"].append(token_index)
            elif touches_argument and _inside(offset, argument_span):
                roles["argument_indices"].append(token_index)
            elif touches_name or touches_argument:
                roles["shared_indices"].append(token_index)
            else:
                roles["other_indices"].append(token_index)
        loss_mask = [True] * len(token_ids)
        arg_range = arg_range_fn(tokenizer, token_ids)
        localization_pass = arg_range is not None
        if arg_range is not None:
            for token_index in range(*arg_range):
                loss_mask[token_index] = False
        cases.append(
            {
                "case_id": f"canonical-{case_index:02d}",
                "turn_index": case_index % 4,
                "raw_call": raw_call,
                "token_ids": token_ids,
                "offset_mapping": offsets,
                "loss_mask": loss_mask,
                "localization_pass": localization_pass,
                **roles,
            }
        )
    return {
        "checkpoint": checkpoint,
        "tokenizer_sha256": hashlib.sha256(vocab_payload).hexdigest(),
        "case_count": len(cases),
        "local_files_only": True,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer

        from areno.api.agentic import _tool_call_arg_token_range
    except ImportError as exc:
        raise RuntimeError("capture requires the authorized AReno model environment") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("exact offset capture requires a fast tokenizer")
    fixture = build_fixture(
        tokenizer,
        str(args.checkpoint_path),
        _tool_call_arg_token_range,
    )
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

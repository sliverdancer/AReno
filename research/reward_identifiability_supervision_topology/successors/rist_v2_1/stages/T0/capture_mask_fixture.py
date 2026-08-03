"""Capture canonical fixtures through the production name-only mask."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any, Callable
import re

TOOL_NAMES = ("scan_registry", "request_hint", "verify_route", "submit_route")
CODE_VALUES = tuple(f"r{index}-{index * 7919:08x}" for index in range(8))
MASK_IMPLEMENTATION = "areno.api.agentic._tool_call_name_only_loss_mask"
_SAFE_TOKENIZER_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
_WEIGHT_SUFFIXES = {".bin", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}
_NAME_FIELD = re.compile(r'"name"\s*:\s*("(?:\\.|[^"\\])*")')
_ARGUMENTS_FIELD = re.compile(r'"arguments"\s*:\s*')


def _overlaps(offset: tuple[int, int], span: tuple[int, int]) -> bool:
    return offset[0] < span[1] and span[0] < offset[1]


def _inside(offset: tuple[int, int], span: tuple[int, int]) -> bool:
    return span[0] <= offset[0] and offset[1] <= span[1]


def tokenizer_snapshot_manifest(root: Path) -> dict[str, Any]:
    """Hash an isolated tokenizer-only snapshot without opening weight files."""

    if not root.is_dir():
        raise ValueError("checkpoint path must be an isolated tokenizer directory")
    rows = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"tokenizer-only snapshot may not contain symlinks: {relative}")
        if path.suffix.lower() in _WEIGHT_SUFFIXES:
            raise ValueError(f"tokenizer-only snapshot contains model weights: {relative}")
        if path.name not in _SAFE_TOKENIZER_FILES:
            raise ValueError(f"unexpected file in tokenizer-only snapshot: {relative}")
        payload = path.read_bytes()
        rows.append(
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not rows or not any(row["path"].endswith("tokenizer_config.json") for row in rows):
        raise ValueError("tokenizer-only snapshot requires tokenizer_config.json")
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"file_count": len(rows), "files": rows, "snapshot_sha256": digest}


def _semantic_spans(text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    name_spans = []
    for match in _NAME_FIELD.finditer(text):
        start, end = match.span(1)
        name_spans.append((start + 1, end - 1))
    argument_spans = []
    decoder = json.JSONDecoder()
    for match in _ARGUMENTS_FIELD.finditer(text):
        try:
            _, consumed = decoder.raw_decode(text[match.end() :])
        except json.JSONDecodeError:
            continue
        argument_spans.append((match.end(), match.end() + consumed))
    return name_spans, argument_spans


def _classify_offsets(
    offsets: list[tuple[int, int]],
    name_spans: list[tuple[int, int]],
    argument_spans: list[tuple[int, int]],
) -> dict[str, list[int]]:
    roles = {
        "name_indices": [],
        "argument_indices": [],
        "other_indices": [],
        "shared_indices": [],
        "masked_boundary_indices": [],
    }
    for token_index, offset in enumerate(offsets):
        touches_name = any(_overlaps(offset, span) for span in name_spans)
        touches_argument = any(_overlaps(offset, span) for span in argument_spans)
        inside_name = any(_inside(offset, span) for span in name_spans)
        inside_argument = any(_inside(offset, span) for span in argument_spans)
        if touches_name and inside_name and not touches_argument:
            roles["name_indices"].append(token_index)
        elif touches_argument and inside_argument and not touches_name:
            roles["argument_indices"].append(token_index)
        elif touches_name:
            roles["shared_indices"].append(token_index)
        elif touches_argument:
            roles["masked_boundary_indices"].append(token_index)
        else:
            roles["other_indices"].append(token_index)
    return roles


def build_fixture(
    tokenizer: Any,
    checkpoint: str,
    tokenizer_revision: str,
    mask_fn: Callable[[Any, list[int], list[bool], tuple[str, ...]], list[bool]],
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build 32 cases with actual offsets and the production name-only mask."""

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
        roles = _classify_offsets(offsets, *_semantic_spans(raw_call))
        localization_error = None
        try:
            loss_mask = mask_fn(
                tokenizer,
                token_ids,
                [True] * len(token_ids),
                (raw_call,),
            )
            localization_pass = True
        except ValueError as exc:
            loss_mask = [False] * len(token_ids)
            localization_pass = False
            localization_error = f"{type(exc).__name__}: {exc}"
        cases.append(
            {
                "case_id": f"canonical-{case_index:02d}",
                "turn_index": case_index % 4,
                "raw_call": raw_call,
                "token_ids": token_ids,
                "offset_mapping": offsets,
                "loss_mask": loss_mask,
                "localization_pass": localization_pass,
                "localization_error": localization_error,
                **roles,
            }
        )
    return {
        "checkpoint": checkpoint,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_sha256": hashlib.sha256(vocab_payload).hexdigest(),
        "tokenizer_snapshot": snapshot,
        "mask_implementation": MASK_IMPLEMENTATION,
        "supervision_mode": "name_only",
        "fixture_kind": "canonical_tokenizer_preflight",
        "runtime_response_tokens": False,
        "case_count": len(cases),
        "local_files_only": True,
        "model_weights_present": False,
        "cases": cases,
    }


def build_runtime_fixture(
    tokenizer: Any,
    checkpoint: str,
    tokenizer_revision: str,
    journal_rows: list[dict[str, Any]],
    mask_fn: Callable[[Any, list[int], list[bool], tuple[str, ...]], list[bool]],
    snapshot: dict[str, Any],
    journal_sha256: str,
) -> dict[str, Any]:
    """Select 8 actual response-token rows per turn from a calibration journal."""

    selected: dict[int, list[dict[str, Any]]] = {index: [] for index in range(4)}
    seen = set()
    for row in journal_rows:
        turn_index = row.get("turn_index")
        if turn_index not in selected or len(selected[turn_index]) >= 8:
            continue
        response = row.get("raw_response")
        metadata = response.get("areno") if isinstance(response, dict) else None
        choices = response.get("choices") if isinstance(response, dict) else None
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        tokens = metadata.get("response_tokens") if isinstance(metadata, dict) else None
        if not isinstance(tokens, list) or not tokens or not isinstance(calls, list) or not calls:
            continue
        key = (
            row.get("task_id", row.get("prompt_index")),
            row.get("sample_index", row.get("rollout_seed")),
            turn_index,
        )
        if key in seen:
            raise ValueError(f"duplicate runtime journal row: {key}")
        seen.add(key)
        selected[turn_index].append({"row": row, "calls": calls, "tokens": tokens})
    counts = {str(turn): len(rows) for turn, rows in selected.items()}
    if any(count != 8 for count in counts.values()):
        raise ValueError(f"runtime fixture requires 8 valid responses per turn: {counts}")

    cases = []
    for turn_index in range(4):
        for within_turn, item in enumerate(selected[turn_index]):
            tokens = [int(token) for token in item["tokens"]]
            text = tokenizer.decode(tokens)
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            encoded_tokens = [int(token) for token in encoded["input_ids"]]
            offsets = [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]]
            if encoded_tokens != tokens or len(offsets) != len(tokens):
                raise ValueError("runtime response tokens do not decode/encode exactly")
            raw_calls = tuple(
                json.dumps(call, ensure_ascii=False, sort_keys=True)
                for call in item["calls"]
            )
            localization_error = None
            try:
                loss_mask = mask_fn(tokenizer, tokens, [True] * len(tokens), raw_calls)
                localization_pass = True
            except ValueError as exc:
                loss_mask = [False] * len(tokens)
                localization_pass = False
                localization_error = f"{type(exc).__name__}: {exc}"
            roles = _classify_offsets(offsets, *_semantic_spans(text))
            cases.append(
                {
                    "case_id": f"runtime-t{turn_index}-{within_turn:02d}",
                    "turn_index": turn_index,
                    "source_key": {
                        "task_id": item["row"].get("task_id"),
                        "prompt_index": item["row"].get("prompt_index"),
                        "sample_index": item["row"].get("sample_index"),
                        "rollout_seed": item["row"].get("rollout_seed"),
                    },
                    "raw_call": text,
                    "token_ids": tokens,
                    "offset_mapping": offsets,
                    "loss_mask": loss_mask,
                    "localization_pass": localization_pass,
                    "localization_error": localization_error,
                    **roles,
                }
            )
    base = build_fixture(
        tokenizer, checkpoint, tokenizer_revision, mask_fn, snapshot
    )
    return {
        **{key: value for key, value in base.items() if key != "cases"},
        "fixture_kind": "runtime_calibration_responses",
        "runtime_response_tokens": True,
        "source_journal_sha256": journal_sha256,
        "turn_case_counts": counts,
        "case_count": len(cases),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--runtime-journal", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer

        from areno.api.agentic import _tool_call_name_only_loss_mask
    except ImportError as exc:
        raise RuntimeError("capture requires the authorized AReno model environment") from exc
    snapshot = tokenizer_snapshot_manifest(args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("exact offset capture requires a fast tokenizer")
    if args.runtime_journal is None:
        fixture = build_fixture(
            tokenizer,
            args.checkpoint_id,
            args.tokenizer_revision,
            _tool_call_name_only_loss_mask,
            snapshot,
        )
    else:
        journal_bytes = args.runtime_journal.read_bytes()
        rows = [
            json.loads(line)
            for line in journal_bytes.decode("utf-8").splitlines()
            if line
        ]
        fixture = build_runtime_fixture(
            tokenizer,
            args.checkpoint_id,
            args.tokenizer_revision,
            rows,
            _tool_call_name_only_loss_mask,
            snapshot,
            hashlib.sha256(journal_bytes).hexdigest(),
        )
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

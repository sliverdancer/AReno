"""Validated, fixed-budget turn-credit routing for agentic GRPO.

The public hook remains file-based and experimental. Core agentic code only
preserves response spans; this module owns the research contract, validation,
normalization, and structured diagnostics.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "care.turn_credit.v1"


@dataclass(frozen=True, slots=True)
class TurnCreditSpan:
    """Immutable hook input for one ordered assistant response span."""

    turn_index: int
    kind: str
    response_start: int
    response_end: int
    token_mass: int
    eligible_token_mass: int
    eligibility_mask: tuple[bool, ...]
    rollout_logprobs: tuple[float, ...]
    raw_text: str
    raw_tool_calls_json: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TurnCreditTrajectory:
    """Immutable hook input for one prompt/sample trajectory."""

    trajectory_id: str
    prompt_index: int
    sample_index: int
    terminal_reward: float
    outcome_advantage: float
    spans: tuple[TurnCreditSpan, ...]
    source_record_json: str
    messages_json: str
    trace_json: str


@dataclass(frozen=True, slots=True)
class TurnCreditBatch:
    """Batch passed to ``route_turn_credit``."""

    schema_version: str
    seed: int
    step: int
    trajectories: tuple[TurnCreditTrajectory, ...]


@dataclass(slots=True)
class RoutedTurnCredit:
    """Validated token-level materialization returned to the trainer."""

    loss_masks: list[list[bool]]
    advantages: list[list[float]]
    diagnostics: list[dict[str, Any]]
    selected_tokens: int
    budget_tokens: int


def load_turn_credit_fn(path: str) -> Callable[..., Any]:
    """Load ``route_turn_credit`` from a user-supplied Python file."""

    module_path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(
        f"areno_turn_credit_{module_path.stem}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load turn-credit function from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        fn = module.route_turn_credit
    except AttributeError as exc:
        raise ValueError(
            f"{module_path} must define callable route_turn_credit(batch, *, step, config)"
        ) from exc
    if not callable(fn):
        raise ValueError(
            f"{module_path} must define callable route_turn_credit(batch, *, step, config)"
        )
    try:
        inspect.signature(fn).bind(object(), step=0, config={})
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{module_path} must define callable route_turn_credit(batch, *, step, config)"
        ) from exc
    return fn


def load_turn_credit_config(path: str | None) -> dict[str, Any]:
    """Load an optional JSON object used by the routing hook."""

    if path is None:
        return {}
    config_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load turn-credit config from {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"turn-credit config must be a JSON object: {config_path}")
    return value


def select_ranked_prefix(
    scores: Sequence[float],
    token_masses: Sequence[int],
    budget_tokens: int,
) -> tuple[int, ...]:
    """Select a deterministic ranked prefix without backfilling.

    Ranking is descending by score with stable index tie-breaking. Selection
    stops at the first item that would exceed the budget. Consequently,
    selections are nested as the budget grows and never replace a skipped
    large item with later small items.
    """

    if len(scores) != len(token_masses):
        raise ValueError("scores and token_masses must have equal lengths")
    if isinstance(budget_tokens, bool) or not isinstance(budget_tokens, int) or budget_tokens < 0:
        raise ValueError("budget_tokens must be a non-negative integer")
    for mass in token_masses:
        if isinstance(mass, bool) or not isinstance(mass, int) or mass < 0:
            raise ValueError("token_masses must contain non-negative integers")
    for score in scores:
        if not isinstance(score, int | float) or not math.isfinite(float(score)):
            raise ValueError("scores must contain finite numbers")

    selected: list[int] = []
    used = 0
    for index in sorted(range(len(scores)), key=lambda item: (-float(scores[item]), item)):
        mass = token_masses[index]
        if used + mass > budget_tokens:
            break
        selected.append(index)
        used += mass
    return tuple(selected)


def build_turn_credit_batch(
    *,
    seed: int,
    step: int,
    response_spans: Sequence[Sequence[Any]],
    response_masks: Sequence[Sequence[bool]],
    loss_masks: Sequence[Sequence[bool]],
    rollout_logprobs: Sequence[Sequence[float]],
    rewards: Sequence[float],
    outcome_advantages: Sequence[float],
    reward_records: Sequence[Any],
) -> TurnCreditBatch:
    """Build immutable, response-relative trajectory records for the hook."""

    row_count = len(response_masks)
    aligned = (
        len(response_spans),
        len(loss_masks),
        len(rollout_logprobs),
        len(rewards),
        len(outcome_advantages),
        len(reward_records),
    )
    if any(length != row_count for length in aligned):
        raise ValueError("turn-credit batch rows are misaligned")

    trajectories: list[TurnCreditTrajectory] = []
    for row_index in range(row_count):
        response_mask = list(response_masks[row_index])
        loss_mask = list(loss_masks[row_index])
        logprobs = list(rollout_logprobs[row_index])
        if len(response_mask) != len(loss_mask) or len(response_mask) != len(logprobs):
            raise ValueError("turn-credit token/mask/logprob row is misaligned")
        response_positions = [index for index, selected in enumerate(response_mask) if selected]
        spans = list(response_spans[row_index])
        if sum(int(span.length) for span in spans) != len(response_positions):
            raise ValueError(
                f"turn-credit response spans do not cover row {row_index}: "
                f"span_tokens={sum(int(span.length) for span in spans)} "
                f"response_tokens={len(response_positions)}"
            )

        credit_spans: list[TurnCreditSpan] = []
        response_offset = 0
        for turn_index, span in enumerate(spans):
            length = int(span.length)
            if length < 0:
                raise ValueError("turn-credit response span length must be non-negative")
            positions = response_positions[response_offset : response_offset + length]
            eligibility = tuple(bool(loss_mask[position]) for position in positions)
            span_logprobs = tuple(float(logprobs[position]) for position in positions)
            credit_spans.append(
                TurnCreditSpan(
                    turn_index=turn_index,
                    kind=str(span.kind),
                    response_start=response_offset,
                    response_end=response_offset + length,
                    token_mass=length,
                    eligible_token_mass=sum(eligibility),
                    eligibility_mask=eligibility,
                    rollout_logprobs=span_logprobs,
                    raw_text=str(getattr(span, "raw_text", "")),
                    raw_tool_calls_json=tuple(getattr(span, "raw_tool_calls_json", ())),
                )
            )
            response_offset += length

        record = reward_records[row_index]
        metadata = getattr(record, "metadata", {})
        prompt_index = int(metadata.get("prompt_index", row_index))
        sample_index = int(metadata.get("sample_index", 0))
        trajectory_id = f"p{prompt_index}:s{sample_index}:r{row_index}"
        source_record = getattr(record, "source_record", {})
        messages = getattr(record, "messages", [])
        trace = getattr(record, "trace", [])
        trajectories.append(
            TurnCreditTrajectory(
                trajectory_id=trajectory_id,
                prompt_index=prompt_index,
                sample_index=sample_index,
                terminal_reward=float(rewards[row_index]),
                outcome_advantage=float(outcome_advantages[row_index]),
                spans=tuple(credit_spans),
                source_record_json=_canonical_json(source_record),
                messages_json=_canonical_json(messages),
                trace_json=_canonical_json(trace),
            )
        )
    return TurnCreditBatch(
        schema_version=SCHEMA_VERSION,
        seed=int(seed),
        step=int(step),
        trajectories=tuple(trajectories),
    )


def route_turn_credit_batch(
    *,
    batch: TurnCreditBatch,
    result: Any,
    token_row_lengths: Sequence[int],
    response_masks: Sequence[Sequence[bool]],
    base_loss_masks: Sequence[Sequence[bool]],
    mini_bs: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> RoutedTurnCredit:
    """Validate hook output and map signed span weights to token advantages."""

    if inspect.isawaitable(result):
        raise ValueError("route_turn_credit must be synchronous")
    result_dict = _require_dict(result, "turn-credit result")
    result_rows = result_dict.get("trajectories")
    if not isinstance(result_rows, list) or len(result_rows) != len(batch.trajectories):
        raise ValueError("turn-credit result trajectories must align one-to-one with the input batch")
    row_count = len(batch.trajectories)
    if (
        len(token_row_lengths) != row_count
        or len(response_masks) != row_count
        or len(base_loss_masks) != row_count
    ):
        raise ValueError("turn-credit token rows are misaligned")
    for row_index, row_length in enumerate(token_row_lengths):
        if (
            int(row_length) != len(response_masks[row_index])
            or int(row_length) != len(base_loss_masks[row_index])
        ):
            raise ValueError(f"turn-credit token/mask lengths are misaligned at row {row_index}")

    routed_masks = [list(row) for row in base_loss_masks]
    raw_advantages = [[0.0] * int(length) for length in token_row_lengths]
    diagnostics: list[dict[str, Any]] = []
    selected_total = 0
    budget_total = 0
    row_budgets: list[int] = []

    for row_index, (trajectory, raw_row) in enumerate(
        zip(batch.trajectories, result_rows, strict=True)
    ):
        row = _require_dict(raw_row, f"turn-credit trajectory {row_index}")
        if row.get("trajectory_id") != trajectory.trajectory_id:
            raise ValueError(f"turn-credit trajectory_id mismatch at row {row_index}")
        turn_rows = row.get("turns")
        if not isinstance(turn_rows, list) or len(turn_rows) != len(trajectory.spans):
            raise ValueError(f"turn-credit turns must align with spans at row {row_index}")
        budget_tokens = _require_int(row.get("budget_tokens"), "budget_tokens", minimum=0)
        selected_tokens_declared = _require_int(
            row.get("selected_tokens"),
            "selected_tokens",
            minimum=0,
        )
        audit_calls = _require_int(row.get("audit_calls"), "audit_calls", minimum=0)
        row_diagnostics = row.get("diagnostics", {})
        _ensure_json_serializable(row_diagnostics, "trajectory diagnostics")

        response_positions = [
            index for index, selected in enumerate(response_masks[row_index]) if selected
        ]
        row_selected = 0
        response_offset = 0
        for span, raw_turn in zip(trajectory.spans, turn_rows, strict=True):
            turn = _require_dict(raw_turn, f"turn-credit turn {span.turn_index}")
            if turn.get("turn_index") != span.turn_index:
                raise ValueError(
                    f"turn-credit turn_index mismatch for {trajectory.trajectory_id}"
                )
            weight = _require_finite_number(turn.get("weight"), "weight")
            abstain = turn.get("abstain")
            if not isinstance(abstain, bool):
                raise ValueError("turn-credit abstain must be a boolean")
            sign = _require_int(turn.get("sign"), "sign", minimum=-1, maximum=1)
            expected_sign = 0 if weight == 0.0 else (1 if weight > 0.0 else -1)
            if sign != expected_sign:
                raise ValueError("turn-credit sign must match weight")
            if abstain != (sign == 0):
                raise ValueError("turn-credit abstain must be true exactly when sign is zero")
            confidence = _require_finite_number(turn.get("confidence"), "confidence")
            if confidence < 0.0 or confidence > 1.0:
                raise ValueError("turn-credit confidence must be in [0, 1]")
            turn_diagnostics = turn.get("diagnostics", {})
            _ensure_json_serializable(turn_diagnostics, "turn diagnostics")
            if not abstain and span.eligible_token_mass == 0:
                raise ValueError("turn-credit cannot select a span with zero eligible tokens")

            positions = response_positions[response_offset : response_offset + span.token_mass]
            span_selected = 0
            for local_index, token_index in enumerate(positions):
                eligible = span.eligibility_mask[local_index]
                if abstain:
                    routed_masks[row_index][token_index] = False
                elif eligible:
                    raw_advantages[row_index][token_index] = weight
                    span_selected += 1
            response_offset += span.token_mass
            row_selected += span_selected
            diagnostics.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "seed": batch.seed,
                    "step": batch.step,
                    "trajectory_id": trajectory.trajectory_id,
                    "prompt_index": trajectory.prompt_index,
                    "sample_index": trajectory.sample_index,
                    "turn_index": span.turn_index,
                    "kind": span.kind,
                    "response_start": span.response_start,
                    "response_end": span.response_end,
                    "sign": sign,
                    "weight": weight,
                    "confidence": confidence,
                    "selected_mass": span_selected,
                    "masked_mass": span.token_mass - span_selected,
                    "budget_tokens": budget_tokens,
                    "audit_calls": audit_calls,
                    "terminal_reward": trajectory.terminal_reward,
                    "outcome_advantage": trajectory.outcome_advantage,
                    "hook_diagnostics": turn_diagnostics,
                    "trajectory_diagnostics": row_diagnostics,
                }
            )
        if row_selected != selected_tokens_declared:
            raise ValueError(
                f"turn-credit selected_tokens mismatch for {trajectory.trajectory_id}: "
                f"declared={selected_tokens_declared} actual={row_selected}"
            )
        if row_selected > 0 and budget_tokens == 0:
            raise ValueError(
                f"turn-credit selected tokens require a positive budget for {trajectory.trajectory_id}"
            )
        if row_selected > budget_tokens:
            raise ValueError(
                f"turn-credit selected tokens exceed budget for {trajectory.trajectory_id}"
            )
        selected_total += row_selected
        budget_total += budget_tokens
        row_budgets.append(budget_tokens)

    if selected_total:
        _apply_accumulation_group_normalization(
            raw_advantages,
            routed_masks,
            row_budgets,
            mini_bs=mini_bs,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    return RoutedTurnCredit(
        loss_masks=routed_masks,
        advantages=raw_advantages,
        diagnostics=diagnostics,
        selected_tokens=selected_total,
        budget_tokens=budget_total,
    )


def write_turn_credit_diagnostics(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    """Append validated turn diagnostics to a JSONL artifact."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("a", encoding="utf-8") as handle:
        for record in records:
            _ensure_json_serializable(record, "diagnostic record")
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _apply_accumulation_group_normalization(
    advantages: list[list[float]],
    loss_masks: Sequence[Sequence[bool]],
    row_budgets: Sequence[int],
    *,
    mini_bs: int | None,
    gradient_accumulation_steps: int | None,
) -> None:
    """Match GRPO's pack mean and the engine's accumulation averaging exactly."""

    row_count = len(advantages)
    resolved_mini_bs = row_count if mini_bs is None else int(mini_bs)
    if resolved_mini_bs <= 0:
        raise ValueError("turn-credit mini_bs must be positive")
    packs = [
        list(range(start, min(start + resolved_mini_bs, row_count)))
        for start in range(0, row_count, resolved_mini_bs)
    ]
    resolved_accumulation = (
        len(packs)
        if gradient_accumulation_steps is None
        else int(gradient_accumulation_steps)
    )
    if resolved_accumulation <= 0:
        raise ValueError("turn-credit gradient_accumulation_steps must be positive")

    for group_start in range(0, len(packs), resolved_accumulation):
        group = packs[group_start : group_start + resolved_accumulation]
        group_size = len(group)
        group_budget = sum(
            row_budgets[row_index]
            for pack in group
            for row_index in pack
        )
        if group_budget == 0:
            if any(
                any(loss_masks[row_index])
                for pack in group
                for row_index in pack
            ):
                raise ValueError("turn-credit accumulation group has selected tokens but zero budget")
            continue
        for pack in group:
            pack_selected = sum(
                sum(bool(item) for item in loss_masks[row_index])
                for row_index in pack
            )
            scale = pack_selected * group_size / group_budget
            for row_index in pack:
                for token_index, value in enumerate(advantages[row_index]):
                    advantages[row_index][token_index] = value * scale


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, list):
        value = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in value
        ]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict")
    return value


def _require_int(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"turn-credit {name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            raise ValueError(f"turn-credit {name} must be >= {minimum}")
        raise ValueError(f"turn-credit {name} must be in [{minimum}, {maximum}]")
    return value


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"turn-credit {name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"turn-credit {name} must be a finite number")
    return result


def _ensure_json_serializable(value: Any, name: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"turn-credit {name} must be JSON-serializable") from exc

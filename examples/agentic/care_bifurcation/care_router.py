"""Actual P3 CARe router for the controlled bifurcation pilot.

The router splits unique prompt blocks within one behavior-policy batch:
even prompt indices are calibration-only and odd prompt indices are update
blocks. Both experimental arms execute the same exact counterfactual audits;
only the CARe arm applies the calibrated confidence threshold.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areno.experimental.care import (
    CalibrationTrajectory,
    CalibrationTurn,
    calibrate_block_threshold,
    select_ranked_prefix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task  # noqa: E402


@dataclass(frozen=True, slots=True)
class _TurnEstimate:
    turn_index: int
    action_index: int
    action: int
    token_mass: int
    eligible_token_mass: int
    weight: float
    predicted_sign: int
    confidence: float
    prefix: tuple[int, ...]


def route_turn_credit(batch, *, step: int, config: dict) -> dict:
    """Audit calibration blocks and route signed credit only to update blocks."""

    del step
    mode = str(config.get("mode"))
    if mode not in {"care", "uncalibrated"}:
        raise ValueError("mode must be care or uncalibrated")
    alpha = float(config.get("alpha", 0.20))
    budget_tokens = int(config.get("budget_tokens", 256))
    scorer_seed = int(config.get("scorer_seed", 9102))
    min_calibration = int(config.get("min_calibration_trajectories", 10))
    min_update = int(config.get("min_update_trajectories", 10))
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")

    _validate_unique_prompt_blocks(batch)
    parsed = {
        trajectory.trajectory_id: _parse_trajectory(
            trajectory,
            scorer_seed=scorer_seed,
        )
        for trajectory in batch.trajectories
    }
    calibration_rows = [
        trajectory
        for trajectory in batch.trajectories
        if trajectory.prompt_index % 2 == 0
    ]
    update_rows = [
        trajectory
        for trajectory in batch.trajectories
        if trajectory.prompt_index % 2 == 1
    ]
    if len(calibration_rows) < min_calibration or len(update_rows) < min_update:
        raise ValueError(
            "insufficient prompt blocks for frozen split: "
            f"calibration={len(calibration_rows)} update={len(update_rows)}"
        )

    calibration_trajectories = []
    audit_calls_by_id: dict[str, int] = {}
    for trajectory in calibration_rows:
        estimates, record, valid = parsed[trajectory.trajectory_id]
        if not valid:
            calibration_trajectories.append(
                CalibrationTrajectory(
                    trajectory_id=trajectory.trajectory_id,
                    turns=(
                        CalibrationTurn(
                            turn_index=0,
                            confidence=1.0,
                            predicted_sign=1,
                            oracle_sign=-1,
                            token_mass=budget_tokens,
                        ),
                    ),
                )
            )
            audit_calls_by_id[trajectory.trajectory_id] = 0
            continue
        weights = tuple(int(value) for value in record["weights"])
        threshold = int(record["threshold"])
        turns = []
        audit_calls = 0
        for estimate in estimates:
            oracle_credit = task.exact_credit(
                weights,
                threshold,
                estimate.prefix,
                estimate.action,
            )
            turns.append(
                CalibrationTurn(
                    turn_index=estimate.turn_index,
                    confidence=estimate.confidence,
                    predicted_sign=estimate.predicted_sign,
                    oracle_sign=task.sign(oracle_credit),
                    token_mass=estimate.eligible_token_mass,
                )
            )
            audit_calls += task.counterfactual_terminal_calls(
                estimate.action_index
            )
        calibration_trajectories.append(
            CalibrationTrajectory(
                trajectory_id=trajectory.trajectory_id,
                turns=tuple(turns),
            )
        )
        audit_calls_by_id[trajectory.trajectory_id] = audit_calls

    calibration = calibrate_block_threshold(
        calibration_trajectories,
        alpha=alpha,
        budget_tokens=budget_tokens,
    )
    threshold_for_json = (
        calibration.threshold
        if math.isfinite(calibration.threshold)
        else ("-inf" if calibration.threshold < 0 else "inf")
    )
    common_diagnostics = {
        "router": "care_bifurcation_v1",
        "mode": mode,
        "alpha": alpha,
        "threshold": threshold_for_json,
        "calibration_corrected_risk": calibration.corrected_risk,
        "calibration_trajectory_count": calibration.trajectory_count,
        "calibration_selected_tokens": calibration.selected_tokens,
        "calibration_wrong_sign_tokens": calibration.wrong_sign_tokens,
        "split": "even_prompt_calibration_odd_prompt_update",
    }

    result_rows = []
    for trajectory in batch.trajectories:
        estimates, _record, valid = parsed[trajectory.trajectory_id]
        is_calibration = trajectory.prompt_index % 2 == 0
        if is_calibration:
            selected_turns: set[int] = set()
            row_budget = 0
        elif valid:
            prefix_positions = select_ranked_prefix(
                [estimate.confidence for estimate in estimates],
                [estimate.eligible_token_mass for estimate in estimates],
                budget_tokens,
            )
            selected_turns = {
                estimates[position].turn_index
                for position in prefix_positions
                if mode == "uncalibrated"
                or estimates[position].confidence > calibration.threshold
            }
            row_budget = budget_tokens
        else:
            selected_turns = set()
            row_budget = budget_tokens

        estimate_by_turn = {item.turn_index: item for item in estimates}
        turn_rows = []
        selected_tokens = 0
        for span in trajectory.spans:
            estimate = estimate_by_turn.get(span.turn_index)
            selected = (
                not is_calibration
                and estimate is not None
                and span.turn_index in selected_turns
                and span.eligible_token_mass > 0
            )
            weight = estimate.weight if selected else 0.0
            sign = 0 if weight == 0.0 else (1 if weight > 0.0 else -1)
            if selected:
                selected_tokens += span.eligible_token_mass
            turn_rows.append(
                {
                    "turn_index": span.turn_index,
                    "weight": weight,
                    "sign": sign,
                    "abstain": not selected,
                    "confidence": estimate.confidence if estimate else 0.0,
                    "diagnostics": {
                        "role": "calibration" if is_calibration else "update",
                        "valid_action_turn": estimate is not None,
                    },
                }
            )
        result_rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "turns": turn_rows,
                "selected_tokens": selected_tokens,
                "budget_tokens": row_budget,
                "audit_calls": audit_calls_by_id.get(
                    trajectory.trajectory_id,
                    0,
                ),
                "diagnostics": {
                    **common_diagnostics,
                    "role": "calibration" if is_calibration else "update",
                    "valid_trajectory": valid,
                },
            }
        )
    return {"trajectories": result_rows}


def _validate_unique_prompt_blocks(batch) -> None:
    seen: set[int] = set()
    for trajectory in batch.trajectories:
        if trajectory.prompt_index in seen:
            raise ValueError(
                "P3 CARe pilot requires n_samples=1 so prompt blocks are unique"
            )
        seen.add(trajectory.prompt_index)


def _parse_trajectory(
    trajectory,
    *,
    scorer_seed: int,
) -> tuple[list[_TurnEstimate], dict[str, Any], bool]:
    record = json.loads(trajectory.source_record_json)
    weights = task.validate_task(record["weights"], record["threshold"])
    task_id = str(record["id"])
    estimates: list[_TurnEstimate] = []
    actions: list[int] = []
    action_index = 0
    valid = True
    for span in trajectory.spans:
        if span.kind != "assistant_tool_call":
            continue
        parsed = _single_bit_call(span.raw_tool_calls_json)
        if parsed is None or action_index >= task.HORIZON:
            valid = False
            continue
        prefix = tuple(actions)
        action = parsed
        weight, predicted_sign, confidence = task.cheap_credit(
            task_id=task_id,
            scorer_seed=scorer_seed,
            turn_index=action_index,
            weight=weights[action_index],
            action=action,
        )
        estimates.append(
            _TurnEstimate(
                turn_index=span.turn_index,
                action_index=action_index,
                action=action,
                token_mass=span.token_mass,
                eligible_token_mass=span.eligible_token_mass,
                weight=weight,
                predicted_sign=predicted_sign,
                confidence=confidence,
                prefix=prefix,
            )
        )
        actions.append(action)
        action_index += 1
    if len(actions) != task.HORIZON:
        valid = False
    return estimates, record, valid


def _single_bit_call(raw_calls: tuple[str, ...]) -> int | None:
    if len(raw_calls) != 1:
        return None
    try:
        call = json.loads(raw_calls[0])
        function = call.get("function") or {}
        if function.get("name") != "choose_bit":
            return None
        arguments = json.loads(function.get("arguments") or "{}")
    except (AttributeError, TypeError, json.JSONDecodeError):
        return None
    bit = arguments.get("bit") if isinstance(arguments, dict) else None
    if isinstance(bit, bool) or bit not in (0, 1):
        return None
    return int(bit)

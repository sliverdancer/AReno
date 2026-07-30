"""Engineering reference hook for the experimental turn-credit interface.

This is not the CARe research method. It broadcasts the existing
group-relative outcome advantage to a deterministic no-backfill prefix of
eligible assistant spans. Its purpose is to qualify the hook, fixed-budget
normalization, and diagnostics on the real trainer path before a learned or
audited credit router is introduced.
"""

from __future__ import annotations

from areno.experimental.care import select_ranked_prefix


def route_turn_credit(batch, *, step: int, config: dict) -> dict:
    """Return contract-valid span weights using only the outcome advantage."""

    del step
    budget_tokens = int(config.get("budget_tokens", 64))
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")
    rows = []
    for trajectory in batch.trajectories:
        masses = [span.eligible_token_mass for span in trajectory.spans]
        scores = [abs(trajectory.outcome_advantage)] * len(masses)
        selected = set(select_ranked_prefix(scores, masses, budget_tokens))
        if trajectory.outcome_advantage == 0.0:
            selected.clear()
        turns = []
        for span in trajectory.spans:
            is_selected = span.turn_index in selected and span.eligible_token_mass > 0
            weight = trajectory.outcome_advantage if is_selected else 0.0
            sign = 0 if weight == 0.0 else (1 if weight > 0.0 else -1)
            turns.append(
                {
                    "turn_index": span.turn_index,
                    "weight": weight,
                    "sign": sign,
                    "abstain": not is_selected,
                    "confidence": min(abs(trajectory.outcome_advantage), 1.0),
                    "diagnostics": {"router": "outcome_broadcast_reference"},
                }
            )
        rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "turns": turns,
                "selected_tokens": sum(
                    trajectory.spans[index].eligible_token_mass
                    for index in selected
                ),
                "budget_tokens": budget_tokens,
                "audit_calls": 0,
                "diagnostics": {
                    "router": "outcome_broadcast_reference",
                    "research_method": False,
                },
            }
        )
    return {"trajectories": rows}

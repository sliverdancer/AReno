# Token-matched robustness plan

Status: `RETROSPECTIVE_SENSITIVITY_HARDENED_INDEPENDENT_SCHEDULE_BLOCKED_BY_API`

## Scientific ruling

The current common-support analysis is a useful, outcome-blind sensitivity
analysis, but it is not an independently randomized token-budget experiment.
It compares development performance at equal interpolated cumulative effective
loss tokens after all arms were trained for the same optimizer-step budget.
This weakens the simple explanation that an arm wins only because it exposes
more trainable tokens, but it does not hold fixed the optimizer path, number of
updates, number of sampled trajectories, token identities, or the policy state
at which those tokens were observed. Cumulative trainable tokens are themselves
affected by the supervision treatment and generated trajectories, so the
retrospective token-indexed contrast must not be described as a causal direct
effect with token exposure controlled.

The CPU analysis therefore freezes the claim as
`retrospective_observed_window_equal_token_exposure`. It integrates the
piecewise-linear curve at every observed token knot, requires the overlap to
cover at least half of both each observed window and each arm's total exposure
from token zero, and requires seed-level stability for step, token-AUC, and
token-endpoint interactions. The interval before the first development
checkpoint has known token exposure but no observed outcome and is not
silently included in AUC.

## What the existing analysis can support

If all gates pass, the defensible statement is: the supervision-topology
interaction keeps its direction when the same step-matched runs are reindexed
over a broad common interval of realized trainable-token exposure. It cannot
support: the interaction would remain under a separately executed equal-token
training budget.

## Independent schedule required for the stronger claim

After the three-seed step-matched pilot, select one common token target using
only realized token exposure, never task success. Freeze it before powered
runs. Execute a new paired schedule from the original checkpoints and seeds;
do not continue or segment the step-matched runs. Report target, exposure
immediately before the terminal step, realized terminal exposure, overshoot,
optimizer steps, trajectories, sampled response tokens, and span-specific
trainable tokens. Analyze the independent schedule separately and require the
same interaction direction and seed-level stability as the step schedule.

No single comparison can simultaneously hold optimizer steps, trajectories,
and effective loss tokens fixed when the treatment changes token density.
Accordingly, the paper should present step-matched and independently
token-budgeted estimands as complementary interventions, not treat either as a
perfect adjustment for the other.

## Minimum public API change

AReno currently exposes `--max-steps` but no token-aware stopping boundary.
The minimum new public surface is:

- `TrainerConfig.max_trainable_tokens: int | None = None`;
- `areno train --max-trainable-tokens INTEGER`;
- a cumulative counter computed from the effective post-mask loss tokens that
  actually enter completed optimizer steps;
- a terminal reason and immutable evidence fields for target, pre-step count,
  post-step count, overshoot, global step, dataset cursor, and RNG identity;
- a checkpoint saved only after the completed terminal optimizer step.

The minimal implementable semantics are `stop_after_first_completed_step_at_or_above_target`.
They necessarily allow at most one step of overshoot, which must be bounded and
reported prospectively. Exact zero-overshoot matching would additionally need
token-aware batch splitting or last-step loss-mask truncation; that is not a
small CLI addition and changes the optimized sample. Abruptly watching
TensorBoard from an external wrapper or restarting from segmented checkpoints
is invalid because it cannot prove optimizer, scheduler, dataset-cursor, and
rollout-RNG continuity.

No public config or CLI is changed in this CPU-only stage.

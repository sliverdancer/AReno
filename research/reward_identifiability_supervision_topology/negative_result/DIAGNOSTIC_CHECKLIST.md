# Reward-resolution diagnostic checklist

Run this checklist before any group-relative tool-use RL experiment that compares
supervision topologies, turn masks, action-span masks, or process-credit methods.

## Required before training

- Freeze the task pool, split boundaries, rollout seeds, sampling parameters,
  and GO/KILL thresholds before model outcomes are inspected.
- Collect calibration trajectories with zero retry and preserve raw responses.
- Report per-task-group strict-success counts.
- Report mixed reward group count per structural cell.
- Report all-fail and all-pass collapse rates.
- Report non-zero-advantage group rate under the exact group-relative objective.
- If multiple model families are used, report common high/low cells and
  cross-model transportability.
- Stop before qualification or training if the frozen GO criterion fails.

## Red flags

- All groups are reward-homogeneous.
- Only one model family has mixed groups.
- A single task or cell determines the entire GO decision.
- Thresholds are changed after observing model outcomes.
- Held-out or sealed data is used to tune the task pool.
- A null training result is interpreted without first reporting reward
  resolution.

## Minimal reporting table

| Field | Required value |
| --- | --- |
| task split SHA | exact SHA-256 |
| rollout seed list | exact list or derivation |
| model revision | exact revision |
| group size | integer |
| groups per cell | integer |
| mixed groups per cell | integer |
| all-fail groups | count and rate |
| all-pass groups | count and rate |
| common high cells | list |
| common low cells | list |
| GO/KILL decision | frozen decision string |

## Interpretation rule

If reward-resolution diagnostics fail, the experiment may still be a valid
evaluation or negative result. It is not valid evidence about which supervision
topology improves training.

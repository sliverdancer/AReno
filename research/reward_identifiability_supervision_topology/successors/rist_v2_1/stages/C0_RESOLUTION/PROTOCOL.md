# C0 direct group-resolution calibration

Status: `CPU_ONLY_FROZEN_BEFORE_MODEL_ACCESS`

For each checkpoint family, C0 runs 32 fixed rollout seeds on each of four
independent tasks in every structural cell. Seeds are partitioned prospectively
into four groups of size eight per task. The observed unit is whether a group
contains both strict rewards zero and one—the exact condition for nonzero
group-relative advantage.

Each cell therefore has 16 independent task-stratified group observations per
split. A cell is:

- `collapsed` when the Wilson upper bound on mixed-group incidence is at most
  0.25;
- `resolved` when the Wilson lower bound is at least 0.50;
- `transition` otherwise;
- `heterogeneous` when fewer than three of four task-level mixed rates agree
  with the cell classification.

Calibration and fresh qualification use different nonces and the same fixed
seeds. Only whole cells whose non-transition label transports exactly may be
selected. Each checkpoint requires at least two collapsed and two resolved
cells. The cross-family training pool is the intersection of Qwen and Gemma
cells with identical transported labels, again at least two per band.

Failure closes the current task pool before training. No individual task or
rollout may be selected by outcome.

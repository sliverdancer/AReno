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

Collection is scheduled with frozen bounded concurrency (Qwen 8, Gemma 4) to
avoid spending roughly fifteen serial GPU-hours on 16,384 possible response
turns. Scheduling does not change the scientific unit: every request seed is a
deterministic hash of rollout seed, task signature, and turn index; output rows
are restored to task/seed order; retry count remains zero. Any worker failure,
missing/duplicate task-seed identity, non-contiguous journal turn sequence, or
result/journal hash mismatch invalidates the entire family/split job.

Failure closes the current task pool before training. No individual task or
rollout may be selected by outcome.

After all four jobs, `finalize_c0.py` reruns artifact validation, verifies both
one-shot qualification ledgers against result and journal bytes, calibrates
each family from the nested collection result, intersects transported labels,
and writes the whole-cell filtered `data/train.jsonl`. No manual JSONL
extraction or hand-assembled resolution map is permitted.

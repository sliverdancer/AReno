# Arbor run report: RIST P1 task-instrument validity

## Result

- Best artifact: commit `7f4c8b5` on branch
  `research/reward-identifiability-supervision-topology`.
- Test score: 8/8 versus initial 8/8; delta 0 because M0 is the initial
  material and already saturates the evaluator.
- Check out with `git switch research/reward-identifiability-supervision-topology`.

## Audit trail

```text
[V] n0 Maximize RIST P1 task-instrument validity [dev=8.0 test=8.0] <== M_best
```

## How understanding evolved

- The initial implementation exposed low, intermediate, and high analytic
  reward resolution while keeping all 32 task cells balanced.
- Train and qualification independently passed the same eight contracts.
- Since the metric is a bounded validity checklist, further search could only
  overfit or change the frozen objective; early stopping is the informative
  result.

## Dev versus test

- Nodes improving development: 0.
- Nodes passing a candidate test gate: 0; M0 itself established the baseline
  test score.
- There was no dev/test disagreement and no candidate search.

## Open direction

The next uncertainty is external validity: actual checkpoints may not realize
the analytic strata. That question belongs to P2 inference qualification, not
additional CPU tuning.


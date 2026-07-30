# Arbor run report: restore executable dataclass-bearing agent loading

## Result

- **Best artifact**:
  `research/structured-action-supervision-v1.1@f96c96544284d65bcb2e8e10ab1d9f2aeda08cd6`
- **Test score**: 4 completed frozen cells out of 4, versus an initial 0 out
  of 4; delta +4.
- **How to check it out**:
  `git checkout research/structured-action-supervision-v1.1`
- The accepted change registers dynamically loaded agent modules under a
  deterministic collision-safe name before execution and rolls back the
  registration if execution fails.

The score measures engineering executability only. It does not convert the
scientifically failed Q1 outcome into a positive result.

## What was tried (audit trail)

```text
Run: Restore executable dataclass-bearing agent loading without changing SAS Q1 scientific variables; maximize completed frozen Q1 cells
Cycle 1/1  |  legend: * root  o pending  ~ running  = executed  V merged  x pruned

[*] n0 Restore executable dataclass-bearing agent loading without changing SA
    [V] n1 Register each dynamically loaded agent module under a collision-safe d [dev=2.0 test=4.0] <== M_best
```

## How understanding evolved

- The v1.0 failure occurred before the first model request, isolating the
  defect to Python 3.12 dynamic-module registration rather than CUDA, model
  loading, or the frozen scientific variables.
- A focused regression with `dataclass(frozen=True, slots=True)` reproduced the
  failure and verified both successful loading and cleanup after execution
  errors.
- The fresh v1.1 test showed that the repair generalizes to all four GPU cells.
  It also revealed a separate scientific failure: the base policy generated no
  valid first tool call, so the intended AF/LF contrast never existed.

## Dev vs. test (overfitting check)

- Nodes that improved **dev**: 1.
- Nodes that passed the **test merge gate**: 1.
- There was no high-dev/low-test candidate. The sole minimal loader fix passed
  its independent frozen execution test. The downstream scientific
  qualification gate failed for a different reason and remains rejected.

## Open directions

Any continuation needs a new high-level instrument formulation that can
establish strict base-policy tool executability before training while
preserving an honest held-out evaluation. Candidate prompt, parser, model, or
sampling changes are not refinements of this consumed run; they require a new
protocol, new freeze, and new authorization. No direction is automatically
opened by this Arbor run.

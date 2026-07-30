# P3-v0.2 postmortem, hypotheses, and frozen qualification protocol

Protocol: `CARE-P3-PILOT-v0.2`

Status: `CPU_REQUALIFICATION_IN_PROGRESS_GPU_NOT_AUTHORIZED`

## Scientific interpretation of v0.1

`CARE-P3-PILOT-v0.1` terminated during the first run before model
initialization or a training update. It is therefore an engineering
executability failure, not negative evidence about CARe's learning effect.
The five unopened runs stay unopened and the archived v0.1 evidence must not
be overwritten or spliced into v0.2.

Root cause: `load_turn_credit_fn` executed the configured Python module without
first registering it in `sys.modules`. Python 3.12 `dataclasses` resolves the
defining module during class decoration, so the configured `care_router.py`
failed at import time.

## Competing engineering hypotheses

| ID | Hypothesis | Prediction | Disproof |
|---|---|---|---|
| E1 | Missing module registration was the sole observed blocker | The production loader imports the dataclass-bearing router on Python 3.10 and 3.12 | Either import still fails |
| E2 | Partial imports can contaminate later runs | Registry state is restored after an import exception | A failed import leaves or overwrites a module entry |
| E3 | Another remote lifecycle or dependency incompatibility remains | v0.2 reaches model initialization and the first finite update | A new crash occurs before a valid update |
| E4 | The CARe signal is degenerate even when executable | At least one CARe run has nonzero selected and masked token mass with finite metrics | All CARe updates select zero mass or rewards collapse |

Only E1 and E2 are CPU-testable. P3-v0.2 is needed for E3/E4, but it remains
blocked on new explicit GPU authorization.

## Scientific hypotheses after qualification

- H1: at matched audit and trainable-token budgets, CARe reduces
  fixed-denominator wrong-sign gradient mass relative to uncalibrated local
  credit.
- H2: the reduction remains after adding a deterministic sign-preserving
  baseline derived from STAMP/StepOPSD.
- H3: reduced wrong-sign mass improves reward or sample efficiency without an
  unacceptable abstention or wall-clock penalty.
- H4: per-iteration recalibration remains valid enough under on-policy drift;
  otherwise risk control fails under deployment-like training dynamics.

H2 is now the main-conference novelty gate. Failure of H2 kills the method-paper
claim even if H1 passes.

## Frozen v0.2 treatment

The scientific design is unchanged from v0.1:

- arms: `care`, `uncalibrated`;
- seeds: `3101`, `3102`, `3103`;
- model asset: the same hash-verified ModelScope Qwen3-0.6B snapshot;
- dataset seed and task: unchanged;
- one update per run, same arm order, token budget, audit count, and
  hyperparameters;
- stop on the first failed/timeout/non-finite run;
- ceilings: 6 single-GPU hours, 8 instance hours, CNY 60.

The only admissible changes are:

1. register the turn-credit module before executing it;
2. restore prior registry state if execution fails;
3. exercise that exact production loader in CPU preparation and remote
   preflight.

No outcome, arm, seed, task, model, threshold, or estimator changed after
observing v0.1.

## Mechanical gates

CPU passes only if:

- the dataclass-bearing router loads through `load_turn_credit_fn`;
- the loader retains a successful module and cleans a failed partial import;
- the P3 design and artifact tests pass;
- Python 3.12 directly loads the frozen router;
- manifest protocol is exactly `CARE-P3-PILOT-v0.2`;
- generated artifacts still carry `GPU_EXECUTION_NOT_AUTHORIZED`.

GPU remains blocked after CPU pass. With a new explicit authorization, execute
the same six-run order and apply the unchanged v0.1 gate:

- `PASS_P3_EXECUTABILITY_TO_P4` only if all six runs complete, required
  artifacts exist, metrics are finite, reward and CARe routing are
  non-degenerate, audits match, and ceilings hold;
- otherwise stop on the first terminal condition and return
  `KILL_P3_EXECUTABILITY`, `KILL_P3_SIGNAL_DEGENERATE`, or
  `INVALID_P3_PROTOCOL`.

P3 is qualification evidence only. It cannot support efficacy, calibration, or
main-conference claims.

# RIST-v2 D1 model-conditional calibration contract

Protocol: `RIST-D1-v2.0`

Mode: CPU-only estimator and synthetic-fixture validation

## Estimand

For a checkpoint-task pair with latent strict-success probability `p` and
training group size `g`, define conditional reward resolution as:

`M_g(p) = 1 - p^g - (1-p)^g`.

Unlike v1, `p` is not set to the inverse of an enumerated action count. It is
estimated from pre-training model rollouts on development calibration nonces.
Binomial uncertainty is propagated through `M_g`; point estimates alone cannot
qualify a cell.

## Structural-cell selection

Each structural cell must contain at least four independent tasks/nonces.
Individual tasks may not be selected because their observed rewards are
convenient. Cells are summarized from task-level estimates and must later be
validated on fresh qualification nonces.

The CPU estimator labels a task:

- `collapsed` only when the upper mixed-probability bound is at most `0.25`;
- `resolved` only when the lower bound is at least `0.50`;
- `transition` otherwise.

These are development labels, not final P2 thresholds. D2 must prospectively
freeze cell-level replication, target bands, family-specific handling, and the
qualification gate before any model is queried.

## Evidence integrity requirements for future clients

1. Append each raw response atomically before parsing or issuing the next turn.
2. Preserve partial trajectories and server errors as first-class records.
3. Record prompt/completion tokens and GPU memory after every turn.
4. Never retry, repair, or synthesize a scientific call.
5. Use independent maximum-context canaries before qualification opens.

## D1 gate

D1 passes only as an engineering gate when deterministic CPU tests validate
the uncertainty transformation, classification boundaries, and v1 D0
reproduction. It cannot open GPU work or upgrade the paper route.

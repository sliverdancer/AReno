# P0 Gate Decision

Protocol: `CARE-P0-v0.1`
Decision date: `2026-07-29`
Decision: `PASS_NOVELTY_CONDITIONAL_TO_P1`
GPU status: `BLOCKED_PENDING_EXPLICIT_USER_AUTHORIZATION`

## Mechanical result

No audited record satisfies all four frozen direct-substitute properties:
signed turn credit, finite-sample wrong-sign risk control, explicit uncertainty
abstention, and fixed supervision-token-budget routing.

Therefore P0 does **not** trigger `KILL_DIRECT_SUBSTITUTE`.

This is not a novelty claim. It opens only the CPU formalization and
exact-oracle falsification stage. The evidence set includes very recent
preprints and cannot establish exhaustive absence.

## Occupied components

The individual components are already heavily occupied:

- TRACE, SIOP, VPR, MT-GRPO, IRC, and TRIAGE provide competing turn or segment
  credit signals.
- ECHO already performs hard token masking from a provenance trace.
- TACO already frames local positive-credit contamination as a calibration
  problem and softly suppresses risky tokens.
- HCAPO, C3, and CCPO provide hindsight or counterfactual estimators, with C3
  serving as a particularly useful exact-intervention oracle in short,
  cooperative text protocols.
- Conformal risk control supplies finite-sample expected-risk machinery, and
  conformal risk training narrows any generic “train with conformal risk”
  novelty claim.

The remaining method cannot be “threshold TRIAGE/TRACE” or “apply conformal
prediction to turns.” It must produce a distinct, testable result about
trajectory-block risk and iterative on-policy recalibration under a declared
gradient-token budget.

## Required P1 correction

The initial target

> wrong-sign fraction among selected turns

is generally non-monotone as the selection threshold changes. Standard
conformal risk control explicitly does not guarantee risk control for an
arbitrary non-monotone loss.

P1 must replace the primary guaranteed loss with a bounded, nested loss having
a fixed denominator:

> wrong-sign gradient mass per trajectory, normalized by a declared maximum
> gradient mass or token budget.

The conditional wrong-sign rate among selected turns remains a diagnostic
without a conformal guarantee. Coverage and task utility are co-primary
efficiency diagnostics, not part of the risk theorem.

## P1 falsification requirements

P1 opens only if all work remains CPU-only and must:

1. state the trajectory-block exchangeability and frozen-policy assumptions;
2. prove or mechanically verify nested monotonicity of the selected update
   family;
3. show that turn-i.i.d. calibration undercovers when errors cluster within a
   trajectory;
4. show block calibration controls the declared wrong-sign gradient-mass risk
   in the same setting;
5. demonstrate guarantee failure under policy shift and require
   per-iteration recalibration;
6. include an exact counterfactual oracle in a controlled bifurcation
   environment; and
7. return `KILL_INCREMENTAL_ONLY` if the result reduces to a generic CRC
   wrapper without an agentic block/on-policy theorem or diagnostic.

## Authorization boundary

P1 may add pure-Python simulations, tests, JSON/CSV artifacts, and formal
notes. It may not change AReno public configuration or CLI surfaces, download
models, run paid APIs, or start GPU training.

# P3 risk-of-bias audit

Protocol: `CARE-P3-PILOT-v0.1`

| Threat | Control | Residual limitation |
|---|---|---|
| Calibration-label leakage | Disjoint calibration rows are assigned zero update budget | Same batch is only iteration-local; no guarantee under later policy shift |
| Pseudoreplication | Whole trajectories are calibration units; `n_samples = 1` | Ten calibration blocks force a relaxed pilot target `alpha = 0.20` |
| Audit-cost confounding | Both arms execute identical exact audits | Pilot does not compare against spending those calls on ordinary rollouts |
| Supervision-density confounding | Fixed 256-token denominator for every update trajectory | Selected token counts still differ by design; compute is not sparse |
| Invalid-call selection bias | Invalid trajectories stay in denominators and abstain | A high invalid rate can make the task unusable |
| Post-hoc threshold tuning | Scorer, confidence values, alpha, split, and seeds are frozen | Synthetic scorer noise is not representative of a learned real-task scorer |
| Seed cherry-picking | Exactly three declared seeds; all outcomes retained | Three seeds estimate executability, not scientific effect size |
| Asset drift | Full-file hash verification after ModelScope download | ModelScope exposes only mutable `master`, so branch name alone is insufficient |
| Outcome overclaim | One-step rewards are explicitly pre-update diagnostics | No downstream learning conclusion is possible |
| Toy-task external validity | Exact enumeration makes the mechanism falsifiable | Any positive result requires later real-task transfer |

The pilot is intentionally biased toward detecting implementation failure. It
is not powered or designed to estimate a publishable method effect.

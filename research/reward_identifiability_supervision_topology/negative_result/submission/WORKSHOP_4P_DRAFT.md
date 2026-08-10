# Workshop 4-page draft

Working title:

`Before Credit Assignment: Reward-Resolution Collapse in Tool-Use RL`

## Core message

Credit-assignment methods decide how to distribute learning signal across
turns. Reward-resolution diagnostics decide whether the sampled rollout group
contains a learning signal to distribute at all.

## Evidence snapshot

| Lineage | Trajectories | Failure mode | Common high cells |
| --- | ---: | --- | ---: |
| RIST-C0-v3.1 | 2,048 | all-fail collapse from exact code bottleneck | 0 |
| RIST-C0-v4.0 | 2,048 | all-fail, all-pass, non-transportable mixedness | 0 |

## Figures

- `../figures/reward_resolution_heatmap.svg`
- `../figures/diagnostic_pipeline.svg`

## 4-page structure

1. Motivation: topology supervision needs reward contrast.
2. Diagnostic: mixed groups and common high/low cells.
3. Evidence: two terminal RIST lineages.
4. Takeaway: report reward-resolution before training or credit assignment.

## Desired workshop feedback

- Is the estimability framing clear enough?
- Which public tool-use benchmark is the strongest external audit target?
- Should the diagnostic be presented as a benchmark-validity checklist, a
  training preflight, or both?

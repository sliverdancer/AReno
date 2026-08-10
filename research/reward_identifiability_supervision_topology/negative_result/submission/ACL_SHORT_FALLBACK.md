# ACL/ARR short fallback

Working title:

`Reward-Resolution Collapse in Group-Relative Tool-Use RL`

## Abstract

Group-relative reinforcement learning is widely used for tool-use agents, but
its supervision and credit-assignment claims require reward variation within
sampled rollout groups. We present two terminal multi-turn tool-use case studies
showing that valid task oracles and serving infrastructure can still yield zero
common high-resolution cells across model families. Across 4,096 calibration
trajectories, RIST v3.1 and v4.0 both failed a frozen reward-resolution gate:
v3.1 collapsed to all-fail rewards, while v4.0 exhibited all-fail, all-pass, and
non-transportable mixedness. We propose a compact diagnostic checklist for
reporting mixed reward groups, all-fail/all-pass collapse, and cross-model
transportability before training supervision topologies.

## Short-paper shape

- 4 pages main text.
- One table: v3/v4 terminal evidence.
- One figure: reward-resolution diagnostic pipeline.
- One appendix: formalism and artifact hashes.

## Best ACL framing

This is a negative finding for NLP/tool-use RL methodology, not a new algorithm.
The paper should emphasize interpretability of experimental claims: topology
training experiments are invalid if reward-resolution diagnostics fail.

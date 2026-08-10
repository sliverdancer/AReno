# Venue assessment

Current status: evidence supports a rigorous negative result, not a main-paper
positive method result.

## Strongest submission route

Near-term best fit:

- workshop on agents, tool use, RLHF/RL, or evaluation methodology;
- negative-results / datasets-and-benchmarks / empirical-validity venue;
- internal technical report that can later become a paper section.

The central contribution is a diagnostic validity layer for group-relative
tool-use RL: reward-resolution contrast must be established before training
supervision topologies.

## Main-conference full-paper probability

Low in the current form.

Reasons:

- no positive training effect;
- synthetic RIST-only evidence;
- two terminal instrument lineages can be read as benchmark design failures
  unless the paper contributes a general diagnostic;
- related work in turn/step credit assignment is moving quickly, so novelty must
  be stated as a precondition/estimability result, not as another credit method.

## What would raise the ceiling

CPU-first additions:

1. A crisp formalism connecting group-relative objectives, reward homogeneity,
   and zero effective advantage.
2. A reviewer threat matrix against TRACE, PORTool, TSPO, and adjacent
   step/turn-level credit papers.
3. A small external audit showing that reward-resolution collapse appears in
   public tool-use traces or benchmark subsets, without using held-out data as a
   tuning source.
4. A reusable diagnostic checklist or script that other researchers can apply
   before training.

GPU additions are not currently recommended. If a future GPU experiment is
needed, it should use a new preregistered task family or public environment with
frozen GO/KILL rules, not a v4 retune.

## Stop conditions

Do not pursue the main-conference route if:

- the paper cannot distinguish reward-resolution diagnostics from ordinary
  benchmark debugging;
- related-work audit finds a direct prior paper making the same group-relative
  estimability claim;
- external data access would require looking at sealed/held-out labels before
  freezing the diagnostic.

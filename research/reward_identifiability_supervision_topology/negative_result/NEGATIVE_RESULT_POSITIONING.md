# Negative-result positioning

Working title:

`When Tool-Use Supervision Becomes Unlearnable: Reward Resolution Collapse under Group-Relative Objectives`

## Claim

The defensible claim is not that one tool-call supervision topology is better
than another. The defensible claim is narrower and stronger:

> Under group-relative objectives, action-span supervision effects are only
> estimable when rollout groups contain enough reward contrast. If every rollout
> in a group receives the same terminal reward, then changing which action spans
> are supervised cannot create a useful policy-gradient signal for strict task
> success.

RIST v3.1 and v4.0 are evidence for the failure mode. They show that an
apparently well-controlled multi-turn tool-use benchmark can pass collection,
validation, and deployment integrity gates while still being scientifically
unusable for supervision-topology experiments because reward resolution
collapses.

## Why this is preferable to retuning v4

Retuning v4 after observing calibration would weaken the research record. The
highest-integrity move is to mark v4 terminal and use it as a case study in
instrument validity:

- v3.1 failed because long non-semantic hash-like codes produced all-fail
  collapse.
- v4 removed that obvious artifact with short semantic codes, but still failed
  the cross-model common-high-cell requirement.
- Gemma solved the easiest v4 cell perfectly, which is still low-resolution for
  group-relative learning because all rewards are identical.
- Qwen stayed all-fail across the v4 cells, so the two-model common contrast
  needed for qualification never appeared.

This makes the result a stronger methodological warning than a benchmark-tuning
story.

## Paper target

Current probability assessment:

- Main-conference full paper: low unless we add broader external validation and
  a constructive diagnostic method.
- Findings / negative-results track / workshop: plausible if framed as a
  rigorous instrument-validity study.
- Internal research report: already supported by the evidence.

To move toward a stronger venue, the next work should be CPU-first:

1. Formalize a reward-resolution diagnostic for group-relative tool-use tasks.
2. Audit related work on turn/step credit assignment and explain why resolution
   collapse is a precondition failure rather than another credit-assignment
   method.
3. Add a small, non-result-conditioned external diagnostic study on existing
   public tool-use traces if available without held-out leakage.
4. Keep RIST v3.1/v4.0 as terminal case studies; do not mutate them into a
   positive benchmark.

## Non-claims

The manuscript should not claim:

- that action-span supervision never matters;
- that tool-call RL is generally unlearnable;
- that v4 proves all semantic-code tasks collapse;
- that Qwen/Gemma model quality is the cause;
- that a positive topology effect is absent in settings with genuine reward
  contrast.

The result is about estimability under the tested group-relative setup.

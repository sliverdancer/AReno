# P4 frozen seed-level analysis protocol

Status: `CPU_ONLY_FROZEN_BEFORE_RESULTS`

P4 analyzes complete run bundles only. It never discovers arms, drops failed
runs, substitutes task/rollout counts for training seeds, or changes endpoints
after seeing outcomes.

## Statistical units and contrasts

The independent unit is a training seed within a checkpoint-family/algorithm
block. For each paired seed, using AF, LF, AN, and LN:

- temporal main effect: `(AF + AN - LF - LN) / 2`;
- content main effect: `(AN + LN - AF - LF) / 2`;
- primary interaction: `(AN - AF) - (LN - LF)`.

Catastrophic runs remain in the intention-to-treat endpoint with strict success
zero. A missing run, missing raw-evidence hash, or unaligned arm makes the block
invalid rather than being omitted.

Every raw-evidence hash is recomputed from an explicitly rooted relative P4
run-evidence manifest. That manifest must itself verify the exact 22 required
roles: source/runtime/resolution identity, training raw and reward journals,
metrics and summary, four checkpoint manifests, four development results and
their raw journals, plus the confirmatory result, raw journal, and one-shot
ledger. Every referenced path is relative, unique, present, size-bound, and
SHA256-bound. The analyzer repeats this transitive verification; a syntactically
valid digest or arbitrary archive cannot pass. It also refuses main-track
eligibility when the execution manifest is still a template, lacks execution
authorization, or contains any treatment not marked scientifically ready.

## Step- and token-matched estimands

The primary endpoint is one-shot D4 confirmatory strict success at the common
frozen optimizer-step budget. Development checkpoint success supplies the
learning curve only. Token robustness combines that development curve with the
same runs' outcome-blind common cumulative-trainable-token support from P3. It
reports normalized AUC and its same paired interaction. A token curve with fewer
than two valid observations or no common support is not estimable and blocks
the main route.

The reporting grid includes both boundaries and the 25/50/75 percent interior
points. AUC itself uses every observed arm-specific knot within the common
interval, not a five-point numerical approximation. The interval must cover at
least 50 percent of every arm's observed token range and total cumulative token
exposure; an arbitrarily narrow or heavily left-truncated overlap is not
robustness evidence. Both normalized-AUC and common-support-endpoint
interactions must preserve the step-matched interaction sign. Catastrophic arms
contribute strict success zero throughout their available token curve rather
than their observed development scores.

Token robustness is required within training seeds, not only after averaging.
Token-AUC and token-endpoint interactions each independently satisfy the same
75 percent seed-sign and leave-one-seed-out stability gates, and at least 75
percent of paired seeds have the same nonzero step/AUC/endpoint sign. A stable
block-mean sign with unstable seed-level token contrasts is a failed robustness
gate.

This remains a retrospective sensitivity estimand. It cannot be described as
an independently executed equal-token intervention because optimizer steps,
policy states, sampled trajectories, and token identities are not held fixed.
The stronger claim requires the separately frozen token-budget API and schedule
in `P3_DESIGN/TOKEN_MATCHED_ROBUSTNESS_PLAN.md`.

Resolution-band endpoints are reported separately for mechanism diagnosis.
Their direction may be learned in the three-seed pilot, but any confirmatory
moderation direction and threshold must be frozen before additional seeds.

## Stability and power gates

Three seeds are pilot-only. Main-track eligibility requires, within all four
family/algorithm blocks:

1. at least `max(8, prospective_power_requirement)` paired seeds;
2. absolute mean step interaction at least 0.10;
3. deterministic seed-bootstrap 95 percent interval excluding zero;
4. at least 75 percent of seed-level interactions agree with their block mean
   sign, and every leave-one-seed-out block mean preserves that nonzero sign;
5. the same nonzero sign across all four blocks;
6. token-AUC and common-support-endpoint interactions with the same mean sign,
   their own seed-level stability, at least 75 percent paired step/AUC/endpoint
   sign agreement, and no block sign reversal;
7. catastrophic-run and zero-advantage-run rates each at most 0.10;
8. complete raw evidence and no post-outcome exclusions.

The bootstrap interval is descriptive when seed count is small. It does not
override the prospective power gate.

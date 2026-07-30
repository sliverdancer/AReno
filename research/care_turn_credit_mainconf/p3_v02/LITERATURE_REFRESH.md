# CARe literature refresh and direct-substitute gate

Snapshot date: `2026-07-30`

Decision: `PASS_NOVELTY_CONDITIONAL_WITH_STRONGER_BASELINES`

This is a targeted refresh of the P0 audit, not an exhaustive systematic
review and not a novelty claim. Searches covered current arXiv records,
OpenReview, ACL Anthology, and PMLR for multi-turn agent credit assignment,
wrong-sign or sign-preserving updates, finite-sample control, and abstention.

## Answer first

No inspected work jointly provides all four CARe components:

1. turn- or trajectory-block credit for multi-turn language agents;
2. a finite-sample upper bound on wrong-sign update mass;
3. abstention from updates whose sign risk is not certified; and
4. comparison under fixed trainable-token and audit budgets.

However, CARe can no longer be positioned as the first method concerned with
credit-sign safety. STAMP and StepOPSD explicitly use sign-preserving
advantage shaping. RLCSD anchors token-level modulation to the outcome
advantage. These are strong near-substitutes and mandatory P5 baselines.

## Strongest new or newly elevated comparisons

| Work | Relevant mechanism | Direct substitute? | Required response |
|---|---|---:|---|
| [STAMP](https://arxiv.org/abs/2607.11172) | Provenance credit with sign-preserving modulation | No: it preserves the trajectory sign rather than estimating finite-sample wrong-sign risk or abstaining | Add a matched sign-preserving baseline; do not claim first sign-safe credit |
| [StepOPSD](https://arxiv.org/abs/2605.27140) | Step-level hindsight distillation with sign-preserving shaping and a normalized credit budget | No: no calibrated block-risk guarantee or selective update gate | Compare at matched trainable-token budget |
| [RLCSD](https://arxiv.org/abs/2606.11709) | Contrastive on-policy distillation anchored by outcome advantage | No: reasoning-token setting and no conformal abstention contract | Include as conceptual anchor and, if implementation is portable, a reasoning-domain baseline |
| [TRACE](https://arxiv.org/abs/2607.13988) | Frozen-model state values and TD turn rewards | No: dense estimator without finite-sample sign-risk abstention | Retain as a high-performing dense-credit baseline |
| [DuCA](https://aclanthology.org/2026.acl-industry.74/) | Separately normalized turn/session advantages | No: balances horizons rather than controlling sign error | Cite as multi-horizon normalization; use only where reward channels exist |
| [CAP](https://proceedings.mlr.press/v304/tayebati26a.html) | Conformal risk selection and abstention | No: prediction-time risk management, not gradient credit | Clarify that CARe transfers selective-risk logic to update routing |

## Revised defensible gap

The defensible gap is narrower:

> Under an on-policy, non-exchangeable trajectory stream, can a
> trajectory-block calibration rule abstain from uncertain local credit so as
> to reduce fixed-denominator wrong-sign gradient mass, beyond what is achieved
> by deterministic sign preservation, while matching trainable-token and audit
> budgets?

This is publishable only if the answer is positive against sign-preserving
controls. A result that beats only outcome broadcast or unconstrained local
credit is insufficient for a main-conference method claim.

## Search log

Queries included:

- `multi-turn agent reinforcement learning turn-level credit assignment`
- `wrong-sign gradient credit assignment reinforcement learning`
- `sign-preserving advantage agent reinforcement learning`
- `abstention credit assignment reinforcement learning LLM agent`
- `finite-sample credit assignment abstention`

Inclusion required a primary paper page or proceedings record and a mechanism
relevant to turn credit, sign control, or finite-sample abstention. Blog
summaries, generic RL finite-sample analyses, and abstention benchmarks without
training-time credit routing were excluded from the decision matrix.

## Falsification rule

Return `KILL_DIRECT_SUBSTITUTE` before P5 if a full-text audit finds a method
that already combines blockwise finite-sample wrong-sign control, selective
update abstention, and matched compute/token evaluation in multi-turn agentic
RL. Otherwise retain `PASS_NOVELTY_CONDITIONAL_WITH_STRONGER_BASELINES`.

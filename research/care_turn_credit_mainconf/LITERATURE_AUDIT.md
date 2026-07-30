# Literature and Direct-Substitute Audit

Snapshot date: `2026-07-29`
Status: `PRELIMINARY_PASS_TO_CPU_DESIGN_ONLY`
GPU implication: none; GPU work remains blocked

## Review question

Does prior work already provide a turn-level agentic-RL method that:

1. estimates signed causal or process credit;
2. controls the risk of assigning the wrong sign;
3. abstains from updating uncertain turns; and
4. selects credit under a fixed supervision-token budget?

A match on these four properties, under no stronger assumptions and with public
artifacts, is a direct substitute.

## Search contract

- Date range: `2024-01-01` through `2026-07-29`, with older foundational work
  included when cited by recent methods.
- Domains: LLM reinforcement learning, multi-turn agents, tool use, selective
  supervision, process rewards, uncertainty calibration, conformal risk
  control, and counterfactual credit.
- Include: primary papers or surveys whose algorithm changes reward, advantage,
  or gradient allocation across tokens, steps, turns, or trajectories.
- Exclude: inference-only context pruning; standard non-language-agent RL
  without a transferable credit mechanism; SFT-only masking unless it is a
  direct mechanism control.
- Databases: arXiv API, OpenAlex API, Semantic Scholar API, and targeted
  primary-source web search.

Queries included combinations of:

- `"LLM agent" AND "turn-level credit assignment" AND reinforcement learning`
- `selective supervision AND loss masking AND multi-turn language agent`
- `counterfactual OR hindsight AND turn-level credit`
- `conformal OR calibrated OR abstaining AND credit assignment`
- `risk-controlled AND credit assignment AND language agent`

## Search execution record

| Source | Result |
|---|---|
| arXiv API exact turn-credit query | 5 returned; exact phrase query had low recall |
| arXiv API calibrated/abstaining query | 15 returned; identified TACO and iterative reward calibration neighbors |
| OpenAlex broad credit query | 1,861 candidates; first 50 retrieved, broad ranking had low precision |
| OpenAlex selective-supervision query | 2,374 candidates; first 50 retrieved, broad ranking had low precision |
| Semantic Scholar API | HTTP 429 without an API key; recorded as unavailable, not silently replaced |
| Targeted primary-source search | Located the direct and adjacent methods below |

The broad database counts are discovery counts, not screened-study counts.
Absence from the current shortlist is not evidence of absence.

## Direct and adjacent methods

| Method | What it already occupies | Residual distinction to test | Threat |
|---|---|---|---|
| [MT-GRPO / turn-level reward design](https://arxiv.org/abs/2505.11821) | Turn-level advantage in multi-turn tool use | No declared finite-sample sign-risk control or abstention | High |
| [RAGEN / StarPO](https://arxiv.org/abs/2504.20073) | Stability and uncertainty-aware filtering | Primarily trajectory/action uncertainty, not certified sign-selective routing | Medium |
| [TRACE](https://arxiv.org/abs/2607.13988) | Dense TD credit from frozen-reference gold-answer potentials | CARe may wrap this scorer and abstain when its sign is unreliable | Critical |
| [SIOP](https://arxiv.org/abs/2605.04984) | Verifier-free potential-based turn credit | Could remove CARe's claimed need for answer supervision | Critical |
| [VPR](https://arxiv.org/abs/2605.10325) | Dense verifier-grounded process rewards | Strong direct baseline when intermediate oracles exist | Critical |
| [ECHO](https://arxiv.org/abs/2606.31650) | Source-indexed turn selection and positive-credit routing | Selection is coupled to memory reconstruction, not explicit sign-risk control | High |
| [TACO](https://arxiv.org/abs/2607.07976) | Risk-aware token credit calibration and suppression of positive-credit contamination | Token-tail risk is adjacent to turn-credit sign risk; distinction must survive full-text audit | Critical |
| [Iterative reward calibration](https://arxiv.org/abs/2604.02869) | Empirical calibration of dense per-turn rewards and advantage direction | No preliminary evidence of distribution-free risk control or abstention | Critical |
| [TRIAGE](https://arxiv.org/abs/2606.32017) | Role-typed segment credit using a structured judge | CARe's strongest case is controlling the judge's sign error rather than inventing another role scorer | Critical |
| [HCAPO](https://arxiv.org/abs/2603.08754) / [C3](https://arxiv.org/abs/2603.06859) / [CCPO](https://arxiv.org/abs/2603.21563) | Hindsight and counterfactual turn credit | Sparse audits plus amortized risk control may reduce their cost | Critical; full text required |
| [GiGPO](https://arxiv.org/abs/2505.10978) | Critic-free group-in-group step credit | Strong low-overhead baseline | High |
| [Conformal Risk Control](https://arxiv.org/abs/2208.02814) | Finite-sample expected-risk control for monotone losses | Statistical foundation, not an agentic credit method | Enabler |
| [Conformal Risk Training](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6559542f75b4452ebaaf82094c7defb7-Abstract-Conference.html) | End-to-end conformal risk objectives | May narrow the theoretical novelty claim | High |
| [CAP](https://proceedings.mlr.press/v304/tayebati26a.html) | Context-adaptive conformal abstention for LLM/VLM outputs | Output abstention differs from training-time turn-credit abstention | Medium |

The 2026 survey reports 47 recent LLM-RL credit-assignment methods and explicitly
identifies a computation-signal trade-off, heterogeneous action types, and rare
bifurcation points as central challenges:
[From Reasoning to Agentic](https://arxiv.org/abs/2604.09459).
It is useful as a map, not as proof that the shortlist is complete.

## Preliminary gap statement

The defensible working gap is narrower than “select important turns”:

> Existing methods produce or route fine-grained credit, but the inspected
> sources do not yet establish a trajectory-block, finite-sample control on
> wrong-sign turn updates with explicit abstention and matched supervision-token
> budgets under an on-policy distribution.

This remains a hypothesis. It becomes a novelty claim only after full-text and
code audit.

## Novelty kill tests

Return `KILL_DIRECT_SUBSTITUTE` if any public method:

- controls false positive/negative turn-credit sign risk with an equivalent
  statistical guarantee;
- uses abstention/masking under an equivalent token budget;
- handles on-policy distribution shift without stronger oracle assumptions;
- and demonstrates the mechanism on comparable agentic benchmarks.

Return `KILL_INCREMENTAL_ONLY` if CARe reduces to:

- thresholding TRIAGE/TRACE scores;
- applying conformal prediction without a new block/on-policy treatment;
- random or top-k masking at matched density;
- or a wrapper whose gain disappears when environment calls and token budgets
  are matched.

## Full-text P0 output contract

P0 must add a frozen comparison matrix with columns:

- estimator target and granularity;
- source of supervision;
- positive/negative credit behavior;
- abstention behavior;
- calibration guarantee and assumptions;
- environment re-execution cost;
- auxiliary-model cost;
- trainable-token treatment;
- public code/data status;
- benchmark/model/seed evidence;
- exact residual difference from CARe.

Only then may the decision become `PASS_NOVELTY_CONDITIONAL`.

## Limitations of this snapshot

- Semantic Scholar was rate-limited.
- Several 2026 methods are recent preprints and may change.
- HCAPO/C3/CCPO have not yet been fully reproduced locally.
- Search results do not prove novelty.
- No AI-rendered schematic was generated because the optional external
  rendering key was unavailable; the protocol uses a version-controlled
  Mermaid schematic instead.

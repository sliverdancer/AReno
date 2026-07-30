# Main-Conference Literature Addendum

Search date: `2026-07-30`  
Purpose: direct-substitute check for the proposed main-track framing  
Status: `PASS_EMPIRICAL_ROUTE_CONDITIONAL`; `METHOD_NOVELTY_UNOPENED`

This addendum does not replace the scoped P0 log. It records new constraints
found while formalizing the main-conference plan.

## Newly emphasized direct neighbors

| Work | Direct constraint on this project |
|---|---|
| [ActFocus](https://arxiv.org/abs/2605.14558) | Already downweights reasoning tokens and redistributes weight among action tokens using token energy. Generic adaptive action-token weighting is occupied. |
| [SOAR](https://aclanthology.org/2026.acl-long.1624/) | Assigns supervision to observation tokens using preceding-action entropy. Observation-aware dynamic supervision is occupied. |
| [MT-GRPO with Iterative Reward Calibration](https://arxiv.org/abs/2604.02869) | Combines multi-turn credit with token-level optimization and empirically calibrated dense rewards. |
| [Rethinking Entropy for Tool Use](https://aclanthology.org/2026.acl-long.1288/) | Uses entropy reduction as sparse and dense supervision for tool behavior. Entropy alone is not a novel adaptive signal. |
| [ToolPRM](https://aclanthology.org/2026.acl-long.855/) | Separates function-name and argument decisions inside structured calls, constraining intra-call novelty claims. |
| [Agentic credit-assignment survey](https://arxiv.org/abs/2604.09459) | Catalogs a large and rapidly expanding token/segment/step/turn credit-assignment space and proposes benchmark/reporting standards. |
| [Proxy state-based evaluation](https://aclanthology.org/2026.acl-industry.87/) | Shows that auditable state-based multi-turn evaluation is itself an active contribution area; our benchmark route needs a distinct supervision-topology focus. |

## Updated novelty decision

Killed now:

- “adaptive action-span selection” as an automatically novel method;
- entropy-, uncertainty-, or action-energy weighting without a discriminating
  mechanism beyond direct prior art;
- function-name versus argument decomposition as a first contribution;
- generic multi-turn credit assignment novelty.

Still conditionally defensible:

- a controlled factorial study that isolates temporal breadth, intra-call
  content, masked-length geometry, and strict protocol validity;
- a supervision-topology benchmark spanning environments, model families, and
  algorithms;
- a later method derived from a replicated failure mode, only after an updated
  direct-substitute audit and faithful baselines.

## Mandatory N1 search after Q1

Search terms and inclusion criteria must be frozen before the search. N1 must
cover action-token weighting, dynamic span selection, counterfactual/hindsight
credit, process rewards, observation supervision, and structured function-call
training. A direct substitute closes the method route without affecting the
truthful diagnostic/benchmark result.


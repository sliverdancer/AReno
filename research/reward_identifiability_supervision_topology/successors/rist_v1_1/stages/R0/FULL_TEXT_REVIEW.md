# R0 Full-Text Reviewer-Threat Review

Protocol: `RIST-R0-v1.1`

Search date: `2026-08-01`

Decision: `PASS_NOVELTY_CONDITIONAL_TO_E0`

## Answer first

No inspected work satisfies all five frozen direct-substitute criteria. The
surviving contribution is narrow: a prospective factorial estimate of whether
pre-training conditional strict-reward resolution moderates the effect of
`AF/LF/AN/LN` policy-gradient eligibility, under both step-matched and
cumulative-trainable-token-matched controls.

This is not evidence that static masking is novel. Turn-level credit, reward
collapse, redundant-step masking, full-trajectory versus final-step handling,
tool-name/argument decomposition, task difficulty, and topology-aware data
selection are all occupied. The paper must present the four masks as a causal
instrument for a conditional interaction, not as a new optimizer.

## Search and audit trail

Five frozen English queries were sent to arXiv, OpenAlex, Crossref, and
Semantic Scholar. The search returned 557 records and 496 deduplicated titles;
the top 150 were retained. arXiv, OpenAlex, and Crossref completed all five
queries. Semantic Scholar completed one and failed four, which is recorded as a
retrieval limitation rather than negative evidence. ACL Anthology full papers
were inspected directly for MatchTIR, ELPO, and ToolPRM.

The eight mandatory papers were all inspectable. The rerun also elevated
A2TGPO, BranPO, ECHO, AT2PO, CIGPO, Signal Reshaping, T2-GRPO, and ProxMO for
full-text threat review. `reviewer_threat_matrix.json` records the frozen fields
and criterion flags. Raw index bytes are hash-bound in `raw/MANIFEST.json`.

The optional external scientific-schematic generator was unavailable because
`OPENROUTER_API_KEY` was not configured. No unpublished prompt was transmitted.
The existing local causal map at
`../../../../figures/causal_identifiability_map.svg` remains the visual aid.

## Strongest threats

### CIGPO: strongest mechanism threat

CIGPO directly diagnoses zero-advantage lock-in under homogeneous group
rewards and injects per-turn information-gain rewards, placing the IG signal on
the last token of every intermediate turn. It therefore supports the mechanism
behind the present study: action-rich traces can still carry no group-relative
gradient when rewards are homogeneous. Its comparison changes reward placement
and reward content; it does not measure base-policy strict-reward resolution as
a moderator, vary full calls versus tool names, use independent seed-level
inference, or estimate the proposed interaction.

Source: https://arxiv.org/html/2607.16244

### Signal Reshaping: strongest masking threat

Signal Reshaping preserves within-group semantic ranking with layered rewards,
reweights assistant-token loss mass using step process scores, and routes some
abnormal failures to either a zero full-trajectory mask or last-step retention.
This occupies the closest all/last-adjacent engineering design. The routing is
deterministic by failure cause rather than an experimental mask assignment; it
does not include name-only cells or estimate reward-resolution-by-mask effects.
Its broader training curves are explicitly single-run comparisons.

Source: https://arxiv.org/html/2605.07276

### MatchTIR and ToolPRM: strongest call-content threats

MatchTIR builds turn rewards from similarity over tool names, parameter names,
and parameter contents and includes an exposure-relevant multi-turn versus
expanded-single-turn comparison. ToolPRM decomposes function calling into
function-name and argument decisions. Neither varies which policy tokens are
eligible for gradients: MatchTIR changes rewards/advantages, while ToolPRM
trains a process reward model and guides test-time beam search.

Sources: https://aclanthology.org/2026.acl-long.549.pdf and
https://aclanthology.org/2026.acl-long.855.pdf

### RC-GRPO, AVSPO, TopoCurate, and ProxMO: strongest resolution threats

RC-GRPO and AVSPO show that homogeneous group rewards suppress useful
advantages. TopoCurate selects structurally informative RL tasks partly to
avoid homogeneous rollouts. ProxMO uses task success rate to modulate gradient
intensity and combines it with step aggregation. None crosses a pre-treatment
resolution measure with fixed AF/LF/AN/LN eligibility.

Sources: https://arxiv.org/html/2602.03025,
https://arxiv.org/html/2605.21125,
https://arxiv.org/html/2603.01714, and
https://arxiv.org/html/2602.19225

### Turn-credit family: baseline obligations

ELPO, TRACE, iterative reward calibration, A2TGPO, BranPO, ECHO, AT2PO, and
T2-GRPO all make turn/step credit more discriminative through trees,
information gain, TD credit, posterior-sensitive reward, environment signals,
adaptive clipping, or redundant-step suppression. These are direct baseline
and discussion obligations. They do not make the supervision-topology effect a
function of an independently measured conditional strict-reward-resolution
moderator.

Sources: https://aclanthology.org/2026.acl-long.504.pdf,
https://arxiv.org/html/2607.13988,
https://arxiv.org/html/2604.02869,
https://arxiv.org/html/2605.06200,
https://arxiv.org/html/2602.03719,
https://arxiv.org/html/2606.29745,
https://arxiv.org/html/2601.04767, and
https://arxiv.org/html/2606.08875

## Reviewer threat matrix summary

| Threat | Occupied contribution | Surviving non-overlap | Consequence |
|---|---|---|---|
| CIGPO | reward-variance collapse plus per-turn last-token reward | pre-treatment moderator crossed with fixed gradient eligibility | mandatory mechanism baseline/discussion |
| Signal Reshaping | group comparability plus full-mask/last-step failure routing | randomized/factorial AF/LF and name-only cells | explicitly distinguish routing from treatment |
| MatchTIR / ToolPRM | turn credit and name/argument decomposition | call-content policy-gradient eligibility | include direct reward/PRM baselines where feasible |
| TopoCurate / ProxMO | topology or task difficulty predicts learning value | conditional mask effect, not data selection | hold task mix fixed and report resolution strata |
| A2TGPO / AT2PO / TRACE family | turn-granular advantage and optimization | static eligibility under the same objective | do not claim turn-level credit novelty |

## Frozen claim and terminology

Use:

> Under group-relative objectives, the effect of action-span supervision may
> be gradient-resolvable only when the base policy induces sufficient
> conditional reward variation. We estimate this moderation prospectively by
> crossing temporal and call-content eligibility while controlling update count
> and cumulative trainable-token exposure.

Do not use unqualified `identifiable`. Prefer `gradient-resolvable` or
`estimable under group-relative objectives`. The theory is about the learning
signal available to the chosen estimator, not formal parameter
identifiability.

## R0 gate

- no single inspected work meets all five direct-substitute criteria;
- all likely direct substitutes had inspectable full text;
- result: `PASS_NOVELTY_CONDITIONAL_TO_E0`;
- main-conference hook: `STAY_DIAGNOSTIC_OPEN_E0_CANARY_PREPARATION`;
- no GPU, model inference, serving, download, or training occurred in R0.

## Required changes to the later research route

1. Add CIGPO and Signal Reshaping as named direct baselines/threats before P3.
2. Keep seed/run as the replication unit; tasks and rollouts are observations.
3. Report non-zero-advantage groups, trainable-token mass, per-span gradient
   norms, first irrecoverable error, name accuracy, and argument accuracy.
4. Run both update-count-matched and cumulative-trainable-token-matched
   comparisons.
5. Do not upgrade to a main-conference claim before an independently seeded
   interaction is practical and its uncertainty excludes the frozen null
   region.

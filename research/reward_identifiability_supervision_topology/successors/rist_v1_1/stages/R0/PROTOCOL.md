# R0 Full-Text Reviewer-Threat Protocol

Protocol: `RIST-R0-v1.1`

Frozen: `2026-08-01`

Status: `FROZEN_UNRUN`

## Review question

Does an inspectable prior work already make substantially the same paper-level
contribution as the proposed study: a prospective causal interaction between
conditional reward resolution and action-span supervision topology in
multi-turn tool-agent learning?

## Proposed contribution under audit

The treatment is the crossed four-cell topology:

- `AF`: all action turns, full calls;
- `LF`: last action turn, full call;
- `AN`: all action turns, tool names only;
- `LN`: last action turn, tool name only.

The moderator is a pre-training, base-policy measure of conditional binary
reward resolution, including the probability that a group of size `G` has
non-constant reward:

`q_G(p) = 1 - p^G - (1 - p)^G`.

Primary outcomes are strict task success and the topology-by-resolution
interaction. Stability, gradient-bearing group rate, and sample efficiency are
secondary. Step-matched and cumulative-trainable-token-matched controls are
part of the proposed contribution.

## Sources and search boundary

Search date: `2026-08-01`.

Use at least three independent scholarly indexes or archives from:

- arXiv;
- ACL Anthology;
- OpenAlex;
- Crossref;
- Semantic Scholar.

Search English-language peer-reviewed papers and inspectable preprints through
the search date. Record unavailable or partial full text as unavailable, not as
negative evidence.

## Mandatory full-text targets

1. RC-GRPO (`arXiv:2602.03025`)
2. Advantage Collapse / AVSPO (`arXiv:2605.21125`)
3. TopoCurate (`arXiv:2603.01714`)
4. MatchTIR (`ACL 2026.acl-long.549`)
5. ELPO (`ACL 2026.acl-long.504`)
6. ToolPRM (`ACL 2026.acl-long.855`)
7. TRACE (`arXiv:2607.13988`)
8. Multi-Turn RL with Iterative Reward Calibration (`arXiv:2604.02869`)

Add any newly retrieved candidate whose title or abstract jointly mentions at
least two of reward variation/collapse, action or turn supervision, function
name/arguments, task topology, or multi-turn tool learning.

## Extraction fields

For every included work record:

- publication status and artifact availability;
- environment, checkpoint, algorithm, and replication unit;
- reward granularity and advantage construction;
- supervised spans or credit units;
- whether all/last and full/name are manipulated;
- whether reward resolution is measured before treatment;
- whether a topology-by-resolution interaction is estimated;
- step/token-mass controls;
- strict success, stability, and sample-efficiency outcomes;
- closest reviewer objection and exact non-overlap.

## Frozen direct-substitute rule

Return `KILL_DIRECT_SUBSTITUTE` if one inspectable work, or one explicitly
unified paper contribution, already satisfies all of:

1. multi-turn tool-agent learning;
2. reward resolution/variation is a designed or measured moderator;
3. supervision breadth or call-content eligibility is experimentally varied;
4. the moderator-by-supervision interaction is estimated or is the central
   benchmark claim;
5. the evidence addresses strict success, stability, or sample efficiency.

Do not trigger KILL merely because separate papers occupy individual factors.
Combining adjacent papers is a novelty risk and baseline obligation, not a
direct substitute, unless a single work already unifies the interaction.

## Outcomes

- `PASS_NOVELTY_CONDITIONAL_TO_E0`: exact joint interaction survives; open CPU
  preparation for the non-scientific canary.
- `KILL_DIRECT_SUBSTITUTE`: close the successor route.
- `BLOCKED_FULL_TEXT_OR_ARTIFACT`: a likely direct substitute cannot be
  inspected sufficiently to decide.

Any outcome is terminal for R0. No GPU, serving, inference, model download, or
training is authorized by this protocol.

## Visual disclosure

The optional third-party schematic generator is unavailable because
`OPENROUTER_API_KEY` is not configured. The review may reuse the existing
locally stored causal map at `../../../../figures/causal_identifiability_map.svg`;
no unpublished prompt may be sent to a third party.

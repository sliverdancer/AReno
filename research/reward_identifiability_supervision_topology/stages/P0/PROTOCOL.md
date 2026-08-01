# P0 Direct-Substitute Review Protocol

Protocol: `RIST-P0-v1.0`

Frozen: `2026-08-01`

## Question

Has prior work already established or benchmarked how task/reward resolution
modulates the effect of temporal and content-level action supervision in
multi-turn tool-agent training?

## Concepts and search families

1. reward diversity, reward resolution, homogeneous groups, advantage collapse;
2. multi-turn tool calling, long-horizon agents, credit assignment;
3. all-turn versus last-turn or action-span masking;
4. tool-name versus argument-level supervision;
5. task difficulty, action dependency, benchmark calibration, sample efficiency.

## Sources

- arXiv API and paper pages;
- OpenAlex works API;
- Crossref works API;
- ACL Anthology;
- OpenReview;
- Semantic Scholar when rate limits permit;
- backward and forward citation chaining from direct threats.

## Inclusion

English primary research from 2023 through the search date concerning language
agents or LLM RL. Earlier foundational work is included only when cited by a
directly relevant paper.

## Exclusion

Robotics-only action learning, generic RL without language/tool agents,
unsupported secondary summaries, and papers that report only final benchmark
scores without a relevant task/reward/supervision analysis.

## Direct-substitute rule

Return `KILL_DIRECT_SUBSTITUTE` if a readable work already treats reward or task
resolution as a designed factor and estimates its interaction with either
temporal action-span breadth or call-content retention in multi-turn tool-agent
training, with strict outcome, stability, or sample-efficiency evidence.

Return `BLOCKED_FULL_TEXT_OR_ARTIFACT` if title/abstract evidence indicates that
rule may be met but the methods/results cannot be inspected.

Otherwise return `PASS_NOVELTY_CONDITIONAL`, explicitly treating reward-collapse
methods, turn-level credit methods, name/argument scoring, and task-difficulty
generation as occupied adjacent components.

## Screening record

Record search strings, source counts, duplicates, title/abstract exclusions,
full-text decisions, direct-threat rationale, and unavailable sources. Do not
interpret an unavailable source as zero evidence.


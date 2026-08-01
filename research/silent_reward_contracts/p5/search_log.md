# P5 external-validation search log

Protocol: `ARCA-P5-EXTERNAL-v0.1`

Freeze date: `2026-08-01`

## Discovery sources

The landscape screen uses complementary primary-source channels: official
GitHub repositories, project documentation, and paper records on arXiv or
OpenReview. Search-engine results are discovery aids only; inclusion and every
technical claim must be checked against an official source.

## Frozen discovery queries

| ID | Channel | Query |
|---|---|---|
| P5-Q1 | web/GitHub | `agentic reinforcement learning framework multi-turn tool use GitHub official` |
| P5-Q2 | web/GitHub | `THUDM slime agentic RL framework reward multi turn` |
| P5-Q3 | web/GitHub | `RAGEN agentic reinforcement learning GitHub` |
| P5-Q4 | web/GitHub | `rllm reinforcement learning agents GitHub` |
| P5-Q5 | arXiv | `agentic reinforcement learning reward validation multi-turn framework` |
| P5-Q6 | OpenReview | `agentic reinforcement learning reward multi-turn tool` |

## Audit order

The frozen audit order is `slime`, `Agent-R1`, `RAGEN`, `rLLM`, then
`Agent-Lightning`. For each system, inspect the pinned revision, record all
audited boundaries, reproduce qualifying cases when possible, and retain a
negative inspection record when no case qualifies.

No candidate may be replaced because its inspection is negative. Search logs,
official source URLs, repository pins, inspected paths, and file hashes are
part of the final artifact.

## Executed source results

The registered queries were executed on `2026-08-01`. Official repository
results anchored all five systems. Matching primary paper records were found
on arXiv for Agent-R1 (`2511.14460`), RAGEN (`2504.20073`), and Agent
Lightning (`2508.03680`). No matching rLLM primary paper was found by the
registered title query, so only its official repository and documentation are
used as system sources. OpenReview supplied adjacent multi-turn reward and
reasoning-collapse studies, not a direct substitute.

The audit retained all five candidates: two qualifying cases and three bounded
negative inspections. Exact source paths and hashes are recorded in
`upstream_manifest.json`; technical outcomes are in `artifacts/`.

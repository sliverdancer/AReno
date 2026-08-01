# P0 search log

Protocol: `ARCA-CPU-AUDIT-v0.1`

Search date: `2026-08-01`

Scope: title/abstract and official-artifact direct-substitute screen. This is a
reproducible novelty gate, not a claim that every unpublished or unindexed work
has been exhausted.

## Sources queried

1. arXiv web index: agentic RL reward contract, reward validation, multi-turn
   reward, tool-use reward hacking, credit assignment, and reward calibration.
2. OpenReview: multi-turn agentic RL, reward variance, reward hacking, and
   framework papers.
3. Official framework documentation and repositories: veRL, OpenRLHF, AReaL.
4. Cross-source exact-phrase search for `reward contract` combined with
   `agentic reinforcement learning`.

## Frozen queries

| ID | Query |
|---|---|
| Q1 | `agentic reinforcement learning reward contract validation multi-turn tool use` |
| Q2 | `reward function verification agentic reinforcement learning trace validation` |
| Q3 | `agentic RL reward hacking benchmark multi-turn agents` |
| Q4 | `reinforcement learning reward pipeline bugs reproducibility audit` |
| Q5 | `multi-turn tool calling reward function schema verl official` |
| Q6 | `OpenRLHF multi-turn reward function agent official` |
| Q7 | `AReaL agentic RL reward official docs` |
| Q8 | `automated reward function validation reinforcement learning LLM agent` |

## Screening rule

A work is a direct substitute only if its paper and maintained artifact jointly
provide all four elements:

1. executable contracts across at least two independent agentic-RL systems;
2. typed trace/reward checks plus semantic reward/advantage/mask checks;
3. registered natural or mutation fault fixtures with structured outputs; and
4. a frozen development/held-out framework transfer evaluation.

Works satisfying only reward quality, reward hacking, credit assignment,
training recipes, or a single framework's extension API are adjacent rather
than direct substitutes.

## Framework pins without source inspection

The following `git ls-remote <url> HEAD` results were recorded before opening
P1/P2. Recording a commit does not count as inspecting held-out source.

| Role | Repository | Frozen HEAD |
|---|---|---|
| development candidate | `https://github.com/volcengine/verl.git` | `e9618406de5bad40041d7612554e465ec2003ec1` |
| held-out candidate | `https://github.com/OpenRLHF/OpenRLHF.git` | `bc71bb19464aca306b33080b2d2bb45d154e2f49` |
| optional replication | `https://github.com/inclusionAI/AReaL.git` | `62f955c5e0388aebc6fd58c5ad3f8fcf9d7384b8` |

## Reproducibility limitation

Search-engine ranking is mutable. Stable paper and documentation URLs, exact
queries, screening criteria, and frozen repository commits are therefore the
auditable record. No paid API, model inference, or generated-paper screening
was used.

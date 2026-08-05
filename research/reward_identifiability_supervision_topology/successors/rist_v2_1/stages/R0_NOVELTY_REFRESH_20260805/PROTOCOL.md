# R0 Incremental Full-Text Novelty-Threat Protocol

Protocol: `RIST-R0-REFRESH-v2.0`

Frozen: `2026-08-05`

Status: `FROZEN_BEFORE_FINAL_ADJUDICATION`

## Purpose and inheritance

This is an incremental refresh of `RIST-R0-v1.1`, whose search closed on
2026-08-01. It inherits the prior contribution, extraction fields, and direct-
substitute rule without amendment. Discovery queries were run before this file
was written; no candidate had received a final five-criterion adjudication.
The protocol is therefore a transparent audit contract, not a claim of
prospective registration.

## Review question

Does any inspectable work available by 2026-08-05 already make substantially
the same paper-level contribution: a prospective causal interaction between
conditional reward resolution and action-span supervision topology in
multi-turn tool-agent learning?

The treatment remains the crossed four-cell topology:

- `AF`: all action turns, full calls;
- `LF`: last action turn, full call;
- `AN`: all action turns, tool names only;
- `LN`: last action turn, tool name only.

The moderator remains a pre-training base-policy measure of conditional strict-
reward resolution. Primary evidence remains strict task success and the
topology-by-resolution interaction, with stability, gradient-bearing group
rate, and sample efficiency as secondary outcomes. Both update-count and
cumulative-trainable-token controls remain required.

## Search boundary

- incremental date window: 2026-08-01 through 2026-08-05 inclusive;
- backward targeted recheck: papers named by new full texts or missing from the
  2026-08-01 threat matrix;
- sources: arXiv, ACL Anthology, OpenAlex, Semantic Scholar, and primary paper
  pages discovered by web search;
- unavailable full text is recorded as unavailable, never as negative evidence.

Frozen query concepts combine:

1. multi-turn tool use / tool calling / tool-integrated reasoning;
2. reinforcement learning / GRPO / group-relative optimization;
3. credit assignment / supervision / masking / action tokens;
4. reward variation / collapse / resolution;
5. tool names / parameters / arguments / complete calls.

Mandatory incremental or previously omitted full-text targets are TurnSight
(`2608.04007`), SERL-SQL (`2608.00485`), TACO (`2606.30251`), and PACT
(`2606.16215`). Any retrieved work jointly mentioning at least two of reward
variation, action/turn supervision, function name/arguments, task topology, or
multi-turn tool learning is screened into the candidate set.

## Frozen direct-substitute rule

Return `KILL_DIRECT_SUBSTITUTE` if one inspectable work, or one explicitly
unified paper contribution, satisfies all five criteria:

1. multi-turn tool-agent learning;
2. reward resolution or variation is a designed or measured moderator;
3. supervision breadth or call-content eligibility is experimentally varied;
4. the moderator-by-supervision interaction is estimated or is the central
   benchmark claim;
5. evidence addresses strict success, stability, or sample efficiency.

Separate adjacent papers create baseline and framing obligations but do not
trigger KILL. Missing evidence is `unknown`, not `false`.

## Terminal outcomes

- `PASS_NOVELTY_OPEN_STRICT_V3`: no direct substitute; a new v3 may be created
  only with independent protocol identity, tasks, seeds, entrypoint, and output
  namespace.
- `KILL_DIRECT_SUBSTITUTE`: do not create v3.
- `BLOCKED_FULL_TEXT_OR_ARTIFACT`: do not create v3 until the high-threat work
  can be inspected.

No GPU, model download, inference, training, held-out access, or BFCL task
access is authorized by this protocol.

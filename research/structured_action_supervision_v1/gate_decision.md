# Feasibility Gate Decision

## Decision

`PASS_FEASIBILITY_CONDITIONAL_ACTION_ALIGNED_REDIRECT`

The repository contains the necessary masking mechanisms and a candidate
multi-turn tool environment, so a controlled study is technically feasible.
However, no current example qualifies as the confirmatory scientific instrument.
The next authorized activity is deterministic design and CPU-side qualification.

## Claims killed now

`KILL_ORIGINAL_THREE_ARM_NOVELTY_CLAIM`

The following claims must not appear in a paper, README, abstract, or application:

- static trainable-turn selection is an unoccupied research gap;
- the current modes are the first or most fine-grained static agentic masking
  scheme;
- tool-call argument masking has no precedent;
- the existing three-arm Tic-Tac-Toe comparison measures three distinct
  scientific treatments.

ActFocus directly studies action-focused token-level RL weighting. TRACE,
MT-GRPO, Agent Lightning, and ECHO provide nearby turn- or transition-level
credit-assignment methods. ToolPRM provides adjacent evidence that intra-call
function and argument components can be supervised separately. The exact AReno
factorial may still be useful as a diagnostic, but implementation distinctness
is not novelty.

## Work authorized by this decision

- research documentation and frozen manifests;
- deterministic dataset construction;
- a research-specific runner that rejects malformed or missing calls;
- CPU tests for mask semantics, trajectory validity, split integrity, and metric
  accounting;
- a no-update integrity test for an empty effective loss mask.

## Work not authorized

- GPU training or serving;
- model or dataset downloads;
- public config or CLI changes;
- confirmatory seed consumption;
- performance, convergence, novelty, or generalization claims.

## Next gate

`Q0-DESIGN-AND-INSTRUMENT`

Pass only if all of the following hold:

1. the four factorial arms select distinct, auditable loss masks;
2. missing or invalid model calls are recorded as invalid and never fabricated;
3. reward-causing action arguments are model-generated and retained verbatim;
4. train/dev/test task constraints are disjoint by construction;
5. training and sampling seeds are explicit and recorded;
6. the empty-mask control produces no optimizer or parameter change;
7. all required metrics can be emitted without GPU synchronization in a hot path.

If any condition cannot be satisfied without expanding scope or changing public
interfaces, return `BLOCKED_DECISION_REQUIRED`, identify the exact change, and
ask the user before proceeding.


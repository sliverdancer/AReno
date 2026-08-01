# Research and publication plan

Working title: **Executable Is Not Learnable: Auditing Silent Reward-Contract
Failures in Agentic Reinforcement Learning**

Primary venue: TMLR. A main-conference submission is conditional on stronger
natural external validation and a convincing downstream consequence study;
the current CPU package alone is better framed as an empirical audit/tooling
paper than a method paper.

## Contribution package

1. a six-family executable contract taxonomy spanning trace, reward,
   advantage, treatment, and provenance boundaries;
2. production-path natural cases in three pinned systems;
3. a frozen, CPU-verifiable auditor with clean controls, registered mutations,
   source hashes, and one-shot held-out evaluation;
4. an external natural-validation sample covering five previously unseen
   systems, with two qualifying cases and bounded negative inspections;
5. a bounded dynamic intervention only if a future P4 successor is separately
   frozen and authorized.

## Completed stages

- P0 literature/direct-substitute screen: passed;
- P1 AReno production-boundary reproduction: passed;
- P2 natural cross-system gate: passed with three development failures across
  two systems;
- P3 frozen held-out transfer: passed, plus one AReaL post-freeze replication;
- reviewer red-team and claim ledger: completed;
- GPU harness: v0.2 attempted once and terminated as infrastructure-invalid
  before model loading; no dynamic scientific result exists.
- P5 external natural validation: passed with two cases in two unseen systems,
  zero rule changes, and five framework-level clean controls.

## Remaining stages

### P4 — bounded dynamic consequence

The consumed `ARCA-P4-DYNAMIC-v0.2` attempt failed during reward-module import
before model loading and cannot be resumed. Its intended intervention was to
normalize string/object tool arguments; it did not change the model, prompts,
data, optimizer, or selected trainable tokens. Because the attempt was invalid,
the intervention was not evaluated and the paper remains CPU/source-audit only.
Any v0.3 proposal needs a separate owner decision and a new run root.

### P5 — natural external validation

State: `PASS_P5_EXTERNAL_NATURAL_TO_PAPER`.

P5 was opened by owner decision as an independent CPU evidence gate for the
TMLR route; it did not depend on or retroactively validate P4. Five candidates
were pinned before inspection. Exact upstream function bodies reproduced one
rLLM evaluator-coercion case and one slime evaluation-replay case. Agent-R1,
RAGEN, and Agent Lightning remain bounded negative inspections. No disclosure
was performed.

This satisfies the preregistered case-count gate for paper drafting, but not a
main-conference method-paper gate: the sample is purposive, the clean-control
interval is wide, the slime boundary is not core training, and no valid
downstream consequence experiment exists.

### P6 — paper and artifact review

State: `OPEN_AFTER_P5_PASS`.

Draft the paper around failure contracts and scientific validity, not model
performance. Include all negative inspections, source pins, fixtures, mutation
operators, frozen hashes, artifact schemas, and terminal P4 outcomes. Run an
anonymous artifact audit and a separate paper-review gate before submission.

## Decision schedule

| Gate | GO | KILL / fallback |
|---|---|---|
| P4-v0.2 | not evaluated; infrastructure invalid | retain CPU audit paper; no rerun |
| P5 | passed: two new natural cases in two unseen systems | bounded transfer only; no prevalence claim |
| Paper audit | no critical validity blocker | revise before submission |
| Main-conference positioning | core-training breadth plus valid downstream consequence | TMLR first |

The current strongest defensible route is a TMLR empirical audit/tooling paper.
“Main-conference method paper” remains contingent on evidence beyond this P5
pass and is not the current claim.

# START HERE — Silent Agentic-RL Reward Contracts

Protocol: `ARCA-CPU-AUDIT-v0.1`

Target: TMLR empirical/audit paper; conditional NeurIPS Evaluations & Datasets
or ACL Findings submission only after the frozen CPU gate passes.

Current state: `P5_PASS_EXTERNAL_NATURAL_P4_INVALID_INFRASTRUCTURE`

## Owner goal

Determine whether silent incompatibilities among agent traces, reward adapters,
advantage construction, and loss masks are a cross-system threat to scientific
validity in agentic reinforcement learning, and whether a CPU-only preflight
auditor can detect them before paid training.

The previous CARe P3-v0.2 run and issue #199 mask alias are discovery cases.
They are excluded from held-out evaluation and cannot be reused as independent
replications.

## Binding boundaries

- P0–P3 are CPU-only. Do not download model weights, call paid APIs, serve a
  model, or start GPU training.
- Do not modify public AReno configuration or CLI surfaces without a new owner
  decision.
- Do not silently fix an audited upstream framework. Pin the observed source,
  retain the failing fixture, and evaluate a separate positive control.
- Stop at the first terminal `KILL`, `INVALID`, or `BLOCKED` gate.
- External issues, pull requests, or maintainer disclosures require separate
  authorization.

## Frozen stages

| Stage | Question | Opening condition |
|---|---|---|
| P0 | Is there a direct published or maintained substitute? | Open |
| P1 | Can the AReno failures be reproduced through production contracts? | P0 pass |
| P2 | Are at least three natural failure classes present across two systems? | P1 pass |
| P3 | Does the frozen auditor transfer to a held-out framework? | P2 pass |
| P4 | Does bounded dynamic GPU validation remain necessary and affordable? | P3 pass plus explicit GPU authorization |
| P5 | Do frozen rules reproduce natural failures in unseen systems? | independent owner-opened CPU gate |
| P6 | Is the paper and anonymous artifact package submission-ready? | P5 pass |

## Evidence roots

- Protocol: `PROTOCOL.md`
- Literature: `p0/`
- Production-contract cases: `p1/`
- Cross-framework audit: `p2/`
- Held-out evaluator: `p3/`
- Arbor state: `arbor_run/.arbor/`
- External natural validation: `p5/`

## Current executable work

P0 through P3 are complete. The frozen held-out evaluation passed, and a
post-freeze AReaL replication was recorded separately. P4-v0.2 terminated
before model loading with `INVALID_P4_INFRASTRUCTURE_REWARD_IMPORT`; it may not
be resumed. Start with `README.md` and `p4/GPU_GATE_DECISION_20260801.md`. A
scientifically new GPU attempt requires a separately authorized protocol.
P5 subsequently passed as an independent CPU evidence gate with two natural
cases in two unseen systems. The next authorized work is P6 paper and artifact
review; P5 did not reopen P4 or authorize GPU execution.

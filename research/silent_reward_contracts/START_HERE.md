# START HERE — Silent Agentic-RL Reward Contracts

Protocol: `ARCA-CPU-AUDIT-v0.1`

Target: TMLR empirical/audit paper; conditional NeurIPS Evaluations & Datasets
or ACL Findings submission only after the frozen CPU gate passes.

Current state: `PASS_P3_CPU_AUDITOR_TO_GPU_AUTHORIZATION_REQUEST`

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

## Evidence roots

- Protocol: `PROTOCOL.md`
- Literature: `p0/`
- Production-contract cases: `p1/`
- Cross-framework audit: `p2/`
- Held-out evaluator: `p3/`
- Arbor state: `arbor_run/.arbor/`

## Current executable work

P0 through P3 are complete. The frozen held-out evaluation passed, and a
post-freeze AReaL replication was recorded separately. P4 is now
`AWAITING_NEW_GPU_INSTANCE_AND_EXPLICIT_AUTHORIZATION`; no model download,
serving, or GPU process may start from this state. Start with `README.md`, then
use `p4/GPU_PROTOCOL.md` only after the owner supplies a new instance and a
fresh budget authorization.

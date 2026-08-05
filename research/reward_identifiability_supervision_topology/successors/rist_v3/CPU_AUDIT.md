# RIST v3 CPU protocol-foundation audit

Decision: `PASS_CPU_PROTOCOL_FOUNDATION_GPU_CLOSED`

## What passed

- The 2026-08-05 full-text novelty refresh found no direct substitute under the
  unchanged five-criterion KILL rule.
- v3 has its own protocol namespace, directory, task split names, result-root
  contract, task signatures, and seeds.
- The generated 32-task calibration and 32-task qualification pools are
  balanced across eight cells; the capacity canary is separate.
- Sixty-five task signatures are internally unique and disjoint from the prior
  JSON manifests inspected by the builder.
- Seventy-four numeric seeds (two generation seeds plus 72 rollout seeds) are
  internally unique and disjoint from prior rollout/generation seeds found by
  the builder.
- A clean rebuild reproduced all three task-file SHA-256 values exactly.
- The deployment receipt binds one worktree plus manifest, model, GPU,
  extension, interpreter real path, interpreter bytes, and interpreter version.
- The child command must start with the bound interpreter; there is no PATH
  lookup or fallback interpreter.
- Exact outcome-free replay launched one fake child under a valid receipt and
  launched zero children for wrong-worktree and wrong-interpreter cases.
- Calibration manifest construction does not hash or read the qualification
  task file. Qualification requires an exact calibration admission and a frozen
  map with at least two low and two high cells.
- `8 passed` in the v3-only suite; `30 passed` when combined with the v2.3
  deployment/scientific regressions; `git diff --check` passed.

## What this does not prove

This is an in-session protocol-foundation audit, not the separate independent
review required before GPU use. It does not prove that a real AReno server,
model request collector, terminal finalizer, or clean shutdown path works under
v3. No model or scientific outcome was accessed.

The v3 scientific collector and finalizer are intentionally not inherited by
import from terminal v2.3. They must be ported into the v3 namespace, tested
against the exact deployment gate, frozen, and reviewed before a new GPU
authorization can be requested.

## Gate

- v3 lineage and C0 protocol foundation: `GO`;
- GPU readiness: `NO`;
- next CPU-only stage: freeze the v3 collector/finalizer, then obtain a separate
  independent audit;
- current GPU/model/qualification/training/held-out/BFCL authority: closed.

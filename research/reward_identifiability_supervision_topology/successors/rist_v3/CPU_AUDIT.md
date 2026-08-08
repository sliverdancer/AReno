# RIST v3 CPU collector/finalizer freeze audit

Decision: `PASS_CPU_COLLECTOR_FINALIZER_FREEZE_GPU_CLOSED`

## What passed

- The 2026-08-05 full-text novelty refresh found no direct substitute under the
  unchanged five-criterion KILL rule.
- v3 has its own protocol namespace, directory, task split names, result-root
  contract, task signatures, and seeds.
- The generated 32-task calibration and 32-task qualification pools are
  balanced across eight cells; the capacity canary is separate.
- Sixty-five task signatures are internally unique and disjoint from the prior
  JSON manifests inspected by the builder.
- Seventy-four numeric seeds are internally unique and disjoint from prior
  rollout/generation seeds found by the builder.
- A clean rebuild reproduced all three task-file SHA-256 values exactly after
  forcing LF byte output for cross-platform hash stability.
- The deployment receipt binds one worktree plus manifest, model, GPU,
  extension, interpreter real path, interpreter bytes, and interpreter version.
- The child command must start with the bound interpreter; there is no PATH
  lookup or fallback interpreter.
- Exact outcome-free replay launched one fake child under a valid receipt and
  launched zero children for wrong-worktree and wrong-interpreter cases.
- Calibration manifest construction does not hash or read the qualification
  task file. Qualification requires an exact calibration admission and a frozen
  map with at least two low and two high cells.
- The v3 scientific request collector is self-owned under the v3 namespace and
  does not import v2.3 collector/finalizer code or the v2.1 D4 evaluator.
- The v3 collector records every raw response in a locked journal, enforces zero
  retry, and preserves actual request payloads and per-turn raw responses for
  later audit.
- The v3 terminal finalizer calls the v3 validator and refuses duplicate
  task/seed rows, missing journal rows, non-contiguous raw responses, forbidden
  access-boundary flags, or runtime identity mismatches before writing final
  collection evidence.
- `python -m pytest tests/test_rist_v3_cpu.py -q` passed: 10 passed, 0 failed.
- `python -m py_compile` passed for the v3 C0 builder, manifest builder,
  collector, validator, finalizer, and v3 CPU test file.

## Independent replay review

`INDEPENDENT_REVIEW_20260809.md` records a separate CPU-only review pass over
namespace isolation, access boundaries, collector/finalizer invariants, and
negative tests. It did not use GPU, model serving, BFCL, held-out tasks,
qualification content, training, or prior v2 outcome values.

## Adjacent regression note

A combined run of the v3 suite plus adjacent v2.3 deployment/scientific CPU
regressions produced 19 passed and 3 failed. The three failures are confined to
v2.3 scientific test fixtures whose temporary deployment receipts are not the
canonical byte form required by the v2.3 validator. No v2.3 source file was
changed by this v3 freeze, and this adjacent issue is not scientific input for
v3. It remains a tooling hygiene item if the old v2.3 regression suite needs to
be made green on Windows.

## What this does not prove

No model request was sent. No real AReno server was started. No calibration,
qualification, held-out, BFCL, inference, serving, or training was accessed.
This freeze proves only that the v3 CPU-side request/terminal evidence contract
is internally executable and rejects key contamination or corruption modes.

## Gate

- v3 lineage, C0 protocol foundation, collector, validator, and finalizer:
  `GO`;
- GPU readiness: `NO` until a clean source commit is bound into a fresh GPU
  manifest/receipt and the user explicitly authorizes calibration;
- current GPU/model/qualification/training/held-out/BFCL authority: closed.

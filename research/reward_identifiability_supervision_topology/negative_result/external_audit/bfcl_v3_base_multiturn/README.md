# BFCL v3 Base Multi-Turn External Audit Feasibility

This directory contains CPU-only, outcome-blind metadata for using the public
BFCL v3 Base Multi-Turn split as an external audit target.

Do not place raw BFCL prompt, answer, or function-documentation files here.
`EXECUTION_RECEIPT_STATIC.json` freezes the external-audit execution design but
does not authorize model/API/GPU inference.
`SYNTHETIC_REPLAY_RESULT.json` records CPU-only evaluator/finalizer replay on
dummy records only.
`MINIMAL_CANARY_RUNTIME_RECEIPT_TEMPLATE.json` freezes the next possible
one-task, one-model inference canary but is not executable until all runtime
bindings are filled and separately authorized.
`MINIMAL_CANARY_LOCAL_PREFLIGHT_BLOCKED.json` records the first local fail-fast
preflight, which stopped before any model request.
`MINIMAL_CANARY_TERMINAL_FINALIZER.json` records the remote one-request Qwen3
canary terminal result. The raw response is represented only by hash and is not
tracked here.

Rebuild with:

```bash
python research/reward_identifiability_supervision_topology/negative_result/build_bfcl_feasibility_audit.py
```

Set `BFCL_FEASIBILITY_SOURCE` if the public files are mirrored outside the
default temp directory.

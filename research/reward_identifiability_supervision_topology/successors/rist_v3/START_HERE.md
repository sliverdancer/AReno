# RIST v3 handoff

Status: `CPU_PROTOCOL_FOUNDATION_PASS_GPU_CLOSED`

RIST v3 is a new protocol lineage opened only after
`RIST-R0-REFRESH-v2.0` returned `PASS_NOVELTY_OPEN_STRICT_V3` on 2026-08-05.
It is not a repair, continuation, or selective rerun of terminal C0 v2.3.

Authority order:

1. `INDEPENDENCE_CONTRACT.json` defines the non-reuse boundary.
2. `PROTOCOL.md` defines the scientific and access gates.
3. `stages/C0_RESOLUTION/data/manifest.json` binds fresh tasks and seeds.
4. `stages/C0_RESOLUTION/deployment_entrypoint.py` is the only process-launch
   gate and additionally binds the Python interpreter.
5. `CPU_FREEZE.json` and `CPU_AUDIT.md` delimit what is frozen and what still
   blocks a GPU authority.

The v3 lineage, independent pool, access gate, and interpreter-bound deployment
foundation are established. The scientific request collector/finalizer has not
yet been frozen under v3, and a separate independent review has not yet been
performed. Therefore no GPU, model, serving, inference, training,
qualification, held-out, or BFCL access is currently open.

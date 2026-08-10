# RIST v4 handoff

Status: `CPU_POOL_FREEZE_PASS_GPU_CLOSED`

RIST v4 is a new strict-independent instrument lineage opened after terminal
`RIST C0 v3.1` calibration killed the current transport. v4 is not a repair or
rerun of v3.1: it uses new seeds, task signatures, protocol identifiers, output
roots, and a semantic-code task generator.

Current authority order:

1. `PROTOCOL.md` defines the scientific gate and access boundaries.
2. `stages/C0_RESOLUTION/build_fresh_pool.py` defines the v4 task generator.
3. `stages/C0_RESOLUTION/data/manifest.json` freezes the CPU pool.
4. `stages/C0_RESOLUTION/*deployment*` and collector/finalizer scripts are
   v4-owned runtime gates.
5. `CPU_FREEZE.json` and `CPU_AUDIT.md` record the CPU freeze.

No GPU, model access, serving, inference, training, qualification, held-out, or
BFCL access is currently open. The next admissible step is a future explicit GPU
authorization for a small v4 canary on A800-80GB.

# C0 v2.3 scientific CPU replay audit

Audited commit: `eb22dcf509a83784fd5913c2eb1561ef677a9a32`

An isolated detached worktree at the exact pushed commit replayed the scientific
freeze verifier, all v2.3 pool/deployment/capacity/clean-shutdown/scientific CPU
tests, and Ruff over the new executable surface.

- scientific freeze: 23/23 checks passed;
- focused CPU regressions: 38/38 passed;
- Ruff: passed;
- audit worktree: clean;
- GPU/model/scientific outcome access: none;
- qualification/held-out/BFCL/training access: none.

The broader ordered `test_rist*_cpu.py` run reported 166 passes and one legacy
D2 failure caused by the suite reusing the generic Python module name
`task_generator`; the same legacy test passed alone. This is test-process module
pollution outside the v2.3 executable surface, not evidence about C0 v2.3.

Decision: `PASS_P0_CPU_FREEZE_TO_GPU_REBIND`. The previous A800 endpoint timed
out, so no GPU stage started. A live 80 GB instance must be rebound at the final
frozen commit before any calibration request.

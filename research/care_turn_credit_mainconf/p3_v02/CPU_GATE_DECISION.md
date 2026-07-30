# P3-v0.2 CPU requalification decision

Date: `2026-07-30`

Protocol: `CARE-P3-PILOT-v0.2`

Decision: `PASS_P3_V02_CPU_FREEZE_TO_GPU_AUTHORIZATION_REQUEST`

GPU execution: `NOT_RUN_NOT_AUTHORIZED`

Frozen source commit:
`c96bcf2da464dff36593d43c8d29991d4b998059`

## Verified evidence

The production loader fix was exercised by regression tests covering:

- successful import of a dynamically loaded `@dataclass` module;
- retention of the successfully initialized module in `sys.modules`;
- cleanup/restoration after an import exception;
- loading the actual P3 `care_router.py` during manifest preparation;
- protocol and six-command manifest preservation.

Commands executed from the isolated CARe worktree:

```bash
/mnt/d/Python_Releases/python.exe -m pytest \
  tests/test_care_turn_credit_cpu.py \
  tests/test_care_p3_design_cpu.py \
  tests/test_trainable_turns_ablation_cpu.py \
  tests/test_metrics_cpu.py \
  tests/test_train_cli_config_cpu.py \
  tests/test_protocol_cpu.py -q
```

Result: `113 passed in 16.48s`.

```bash
PYTHONPATH=. /usr/bin/python3 -c \
  "from areno.experimental.care.turn_credit import load_turn_credit_fn; \
  fn=load_turn_credit_fn('examples/agentic/care_bifurcation/care_router.py'); \
  assert fn.__name__ == 'route_turn_credit'; \
  print('PYTHON312_DYNAMIC_LOADER_PASS', fn.__module__)"
```

Runtime: Python `3.12.3`.

Result:

```text
PYTHON312_DYNAMIC_LOADER_PASS areno_turn_credit_care_router
```

Compilation smoke:

```bash
/mnt/d/Python_Releases/python.exe -m compileall -q \
  areno/experimental/care \
  examples/agentic/care_bifurcation \
  tests/test_care_turn_credit_cpu.py \
  tests/test_care_p3_design_cpu.py
```

Result: exit code `0`.

The pytest runtime emitted an existing `requests` dependency-version warning;
it did not fail a test. The repository-wide CPU collection was not claimed
because the Windows runtime has a known unrelated `pty -> termios` collection
incompatibility.

## Gate interpretation

This decision establishes only that the observed v0.1 loader failure is
CPU-reproduced and repaired under Python 3.10 and 3.12, and that the frozen
artifact path remains testable. It is not evidence that a GPU run will reach a
finite update, that CARe has a non-degenerate signal, or that CARe improves
learning.

The next boundary is a new explicit authorization for the exact frozen source
commit, six qualification runs, 6 GPU-hour ceiling, 8 instance-hour ceiling,
and CNY 60 ceiling. Until then, `--execute-gpu-training` must not be used.

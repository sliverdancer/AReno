# P3 frozen GPU gate decision

Protocol: `CARE-P3-PILOT-v0.1`
Execution date: `2026-07-30`
Provider: AutoDL
Hardware: one NVIDIA A800 80GB PCIe
Frozen commit: `dd2bf826771b1e47541edfbee26e59808ebfb1dc`
Decision: `KILL_P3_EXECUTABILITY`
Downstream state: `P4_P5_P6_P7_UNOPENED`

## Frozen execution outcome

The owner explicitly authorized the six-run qualification pilot under the
frozen six-GPU-hour, eight-instance-hour, and CNY 60 ceilings. The executor
started at `2026-07-30T23:34:51+08:00`.

The first declared run, `care-seed-3101`, exited with return code 1 after
8.479367 seconds. The executor applied the frozen stop rule immediately.
The other five runs were not opened, and no seed was repaired, removed, or
selectively rerun.

The failure occurred before checkpoint loading or GPU allocation. Final
observations were zero MiB GPU memory and zero percent GPU utilization. This
run therefore exercised the authorized training entrypoint but did not perform
a model forward pass or optimizer update.

## Observed failure

`load_turn_credit_fn` executed `care_router.py` as a dynamically loaded module.
During the `@dataclass(frozen=True, slots=True)` decoration, Python 3.12
attempted to resolve the class module through `sys.modules`. The dynamic module
was absent from that registry, producing:

```text
AttributeError: 'NoneType' object has no attribute '__dict__'
```

The traceback ends in:

```text
areno/experimental/care/turn_credit.py:86
examples/agentic/care_bifurcation/care_router.py:29
```

This is an instrumentation-loader executability failure, not evidence about
CARe reward, calibration, learning, memory fit, or method efficacy.

## Mechanical decision

`PROTOCOL.md` requires all six commands to exit zero and classifies crashes or
missing artifacts as `KILL_P3_EXECUTABILITY`. The observed return code 1 and
the five unopened runs mechanically satisfy that terminal decision.

The consumed `CARE-P3-PILOT-v0.1` may not be repaired and resumed. Any future
attempt requires a new versioned protocol, a CPU regression reproducing this
loader failure, a new reviewed commit and manifest, and new explicit GPU
authorization.

## Evidence

Local evidence root:

`research/care_turn_credit_mainconf/p3/evidence/autodl_a800_20260730_kill_p3/`

Key hashes:

- frozen manifest:
  `88ba8b1bf3d197f5ba29765d32648b092141d5f012b40548805a6fbe4ab63179`;
- frozen dataset:
  `bdb6fd0f502a9f7a7bb2e2947a09c89d884ffada7364fb92c81d4105c38fa03d`;
- `run_results.json`:
  `8b3ac8a464d47b63b98df42cef6794db8ae8b436db454849e694872a89d21b83`;
- failure bundle:
  `061dab131bd47fc97d40fd0e7718d46f2ffed9cd922e75f5ad563559eccea798`;
- first-run stdout:
  `ac15d726a5eeb5316fa5da02a024b13b3e7131ea3e39186e27c67c02d51d697a`;
- first-run stderr:
  `70e5eb7b103ccbf168383f936cb47a73ee8245c3b698c86f9455ebbf1cb6a1d7`.

No efficacy claim is admissible from this terminal qualification run.

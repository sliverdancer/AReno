# Q1 Bounded GPU Integration Pilot

Decision: `KILL_CURRENT_INSTRUMENT`

The authorized Q1 run stopped at the first prespecified command,
`AF-seed-1101`, before any model request, trajectory, reward, or optimizer step.
The remaining three commands were not started. This is a terminal instrument
failure under `SAS-P0-v1.0`; no efficacy or variance conclusion is available.

## Verified setup

- source snapshot: commit `dd2bf826771b1e47541edfbee26e59808ebfb1dc`;
- GPU: NVIDIA GeForce RTX 4090 D, 24,564 MiB;
- Python 3.12.3, PyTorch 2.8.0+cu128, CUDA runtime 12.8;
- compiled `areno.accel._areno_accel` import passed;
- BF16 CUDA smoke test passed;
- `areno check` reported ready, with only the irrelevant optional
  `flash-attn` warning because Q1 uses native attention;
- 89 SAS/CLI CPU tests passed;
- full CPU collection produced 431 passes and two deployment-environment
  failures: one test expected a Git worktree although the source archive
  intentionally excluded `.git`, and one root-path writability test assumed a
  path was missing.

The ModelScope Qwen3-0.6B snapshot was downloaded and hashed before execution.
The qualification dataset hash matched the frozen manifest. Held-out outcomes
were not read.

## Failure

`load_agent_run_fn` creates the dynamic agent module with
`module_from_spec()` and immediately calls `exec_module()` without registering
the module in `sys.modules`. Python 3.12's `dataclasses` implementation consults
`sys.modules` while processing the frozen agent's
`@dataclass(frozen=True, slots=True)` declaration. The lookup returned `None`,
raising:

```text
AttributeError: 'NoneType' object has no attribute '__dict__'
```

The traceback is preserved in
`attempt_20260730/runs/AF-seed-1101/train.log`. The controller returned 1 after
25.38 seconds and confirmed that no GPU process remained. There was no OOM and
no non-finite optimization, but the required parser/executability, reward,
resource, provenance-completeness, and mask-distinction checks could not be
measured.

## Scientific boundary

The failure is an engineering defect, not evidence for or against AF versus LF.
The frozen Q1 pass gate requires all four commands to complete, so repairing the
loader and selectively continuing would invalidate this attempt. The original
pilot manifest is preserved unchanged, and no completed or partial outcome is
spliced into a later run.

Any continuation must open a new protocol version, add a CPU regression that
loads a dataclass-bearing agent file through the public loader, fix module
registration with failure cleanup, repeat the qualification checks, and freeze
a new four-run manifest. That route is unopened.

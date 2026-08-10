# BFCL external audit CPU-only synthetic replay

Status: `PASS_CPU_ONLY_SYNTHETIC_REPLAY`

This stage verifies the evaluator wiring and terminal finalizer on synthetic
dummy records only. It does not load BFCL raw prompts, possible answers,
function docs, model responses, model weights, GPUs, APIs, held-out data, or
training code.

## What was validated

- Static receipt hash matches `EXECUTION_RECEIPT_STATIC.sha256`.
- Selected task-id hash matches the frozen receipt.
- Public manifest and schema-summary hashes match the frozen receipt.
- Receipt denies model inference, API inference, GPU use, training, and
  held-out/sealed access.
- Strict call evaluation distinguishes:
  - exact pass;
  - wrong tool;
  - wrong argument;
  - empty observed calls;
  - JSON-string arguments normalized to structured arguments.
- Finalizer produces all three reward-resolution categories:
  - mixed reward group;
  - all-pass group;
  - all-fail group.

## Output

`SYNTHETIC_REPLAY_RESULT.json`

Summary:

- synthetic cases: 5
- synthetic groups: 3
- mixed groups: 1
- all-pass groups: 1
- all-fail groups: 1
- model/API/GPU/training used: no

## Scientific interpretation

This result is not external empirical evidence. It only proves that the
pre-inference evaluator/finalizer path is wired and that it fails fast on frozen
receipt/hash mismatches before any model request would be sent.

The next admissible step is to freeze a minimal inference canary runtime receipt.
That canary should be one task and one model first. The full 64-task, two-model,
32-rollout audit remains premature.

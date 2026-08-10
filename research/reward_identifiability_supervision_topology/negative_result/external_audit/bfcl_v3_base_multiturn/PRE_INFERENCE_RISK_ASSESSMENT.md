# BFCL external audit pre-inference risk assessment

Status: `CPU_ONLY_PRE_INFERENCE_ASSESSMENT`

This assessment is made before any model/API/GPU inference. It is intentionally
conservative: the current package freezes what would be executed, but does not
authorize execution.

## Current judgment

Estimated probability that BFCL V3 Base Multi-Turn yields an informative
external audit: `moderate`, approximately `0.55-0.70`, conditional on clean
evaluator integration.

This means the audit is likely to produce useful evidence for the paper in one
of two ways:

1. It finds non-trivial reward-resolution collapse in a public benchmark,
   supporting the claim that reward-resolution diagnostics are not RIST-specific.
2. It does not collapse, which separates RIST v3/v4 as instrument failures and
   shows that the diagnostic can distinguish usable from unusable task pools.

It does not mean a full-paper submission is likely by itself. Venue escalation
still depends on clean external evidence, formal framing, related-work
positioning, and reproducible artifact quality.

## Main risks

- Small open models may fail strict function-call formatting, producing mostly
  all-fail groups. That can still be useful, but it weakens the external-audit
  story if failures are dominated by parser mismatch rather than task
  reward-resolution.
- BFCL's evaluator integration may require code-path assumptions that are not
  captured by the public data files. This must be tested CPU-only before model
  spending.
- A full two-model, 64-task, 32-rollout audit is expensive enough that it should
  not be the first GPU/API step.

## Required gates before any model/API/GPU spend

1. CPU-only evaluator wiring dry-run using synthetic dummy records only.
2. Public-file hash recheck against `PUBLIC_FILE_MANIFEST.json`.
3. Selected-task hash recheck against `SELECTED_TASK_IDS.txt`.
4. Runtime receipt freeze with exact model repo ids, model revisions, tokenizer
   revisions, serving backend, GPU UUID, extension hashes, decoding limits, and
   output roots.
5. One-task, one-model format canary with terminal finalizer enabled.
6. Only if the canary produces parseable evaluator input, open a small
   two-model canary.
7. Only if both canaries are interpretable, consider the full external audit.

## Recommendation

Do not authorize the full 4,096-response BFCL audit yet. The next valid step is
CPU-only evaluator wiring and synthetic replay. If that passes, the first
inference authorization should be a minimal format canary, not the full audit.

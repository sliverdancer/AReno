# P2 Read-Only Instrument Audit

Protocol: `CARE-P2-AUDIT-v0.1`
Audit date: `2026-07-29`
Code changes in this audit: research documents only
GPU training: not performed

## Result

The current worktree is sufficient for the issue #199 static-mask harness, but
it is not sufficient for CARe. The missing path is not another mask mode; it is
preservation of turn boundaries through batch materialization and assignment
of independently signed turn weights before token packing.

## Requirement matrix

| P2 requirement | Current evidence | State |
|---|---|---|
| Assistant-span boundaries exist during rollout | Private `_AgentSample.response_spans` stores kind and length | `PARTIAL` |
| Stable turn offsets reach the trainer | `AgentTrainBatch` drops `response_spans`; only token/mask rows remain | `MISSING` |
| Signed per-turn advantages reach `TrainSequence` | `TrainSequence.advantages` is per-token capable, but agentic GSPO/GRPO materialization broadcasts one group advantage to every selected token | `MISSING` |
| Explicit abstention produces no parameter update | Existing backend skips an explicit all-false loss batch | `PASS_CPU` |
| Trainable-token budget can be enforced | Static masks and token counts exist; no confidence-ranked, no-backfill budget router exists in the runtime | `PARTIAL` |
| Objective normalization is independent of selected length | GRPO divides by selected valid-token count; GSPO constructs sequence ratios/advantages using selected response length | `MISSING` |
| Per-step reward/token artifacts | Reward mean/std plus `trainable_tokens` and `masked_response_tokens` are emitted and the issue #199 collector writes CSV/JSON | `PASS_CPU` |
| Seed provenance | Current dirty worktree adds base, worker, epoch-order, and request seed propagation | `PASS_CPU_UNCOMMITTED` |
| Invalid calls are never synthesized | Structured-action CPU tests reject missing/malformed calls and preserve raw arguments | `PASS_CPU` |
| Counterfactual audit accounting | No first-class environment-branch/call ledger exists in the training batch | `MISSING` |

## Verified tests

Command:

```bash
python -m pytest \
  tests/test_trainable_turns_ablation_cpu.py \
  tests/test_metrics_cpu.py \
  tests/test_care_p1_cpu.py -q
```

Observed result:

```text
15 passed in 12.28s
```

The `requests` package emitted a dependency-version warning. It did not affect
the test result.

## Important normalization finding

The current loss mask changes the normalization population:

- GRPO divides token loss by `layout.valid_count`, which counts selected
  response tokens.
- GSPO averages log ratios and advantages by
  `layout.response_len`, also computed from selected response tokens.

Consequently, two masks with equal raw advantages but different selected-token
counts do not isolate attribution from objective scaling. CARe needs a frozen
budget denominator or an exactly equivalent pre-loss rescaling. The static
three-arm harness must not be used as evidence that this confound is solved.

## Dirty-worktree boundary

The files needed for the minimum runtime change already contain uncommitted
work, including seed propagation, static trainable-turn controls, metrics, and
zero-signal skipping. This audit did not edit or reformat those files. Any P2
implementation must preserve the existing diff and add focused tests around
the overlapping paths.

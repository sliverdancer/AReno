# Minimal P2 Interface Proposal

Status: `AUTHORIZED_IMPLEMENTED_CPU_VERIFIED`
Design goal: expose one experimental batch-level turn-credit hook without
adding another static mask mode.

## Proposed public surface

Add two optional fields only to the online-policy config:

```python
turn_credit_fn_path: str | None = None
turn_credit_config_path: str | None = None
```

Add matching CLI options:

```text
--turn-credit-fn-path FILE
--turn-credit-config-path JSON
```

Both default to `None`, preserving current behavior. The first experimental
implementation is deliberately restricted to agentic GRPO; unsupported
combinations fail before rollout.

GSPO is excluded because its current loss forms one sequence-level ratio and
one sequence-level advantage. It cannot preserve opposite signs for different
turns in the same trajectory. Labeling that path “turn-level credit” would
therefore be scientifically false even if the code accepted the option.

The Python file exposes one batch-level function:

```python
def route_turn_credit(batch, *, step: int, config: dict) -> result:
    ...
```

The hook receives immutable trajectory records with:

- prompt/sample identifiers;
- terminal reward and group-relative outcome advantage;
- ordered assistant spans with response-token start/end offsets;
- span kind, raw turn text/tool call, and existing static eligibility mask;
- rollout log-probabilities and declared token mass.

It returns one row per input trajectory:

- one signed scalar weight per assistant span;
- one explicit sign, confidence, and abstain bit per span;
- selected token mass and fixed budget denominator per trajectory;
- calibration/audit counters and a JSON-serializable diagnostic record.

Malformed length, sign, budget, or row alignment is a hard error before the
backend step.

## Internal changes

1. Preserve `response_spans` in `_AgentTrainRows` and `AgentTrainBatch`.
2. Materialize stable response-token offsets when the spans are appended.
3. Load the hook once in `PolicyOnlyTrainer`.
4. Replace group-advantage broadcast only when the hook is enabled.
5. Rescale routed token advantages per real backend microbatch/gradient-
   accumulation group so the declared budget, not the selected length or
   `mini_bs`, is the objective denominator.
6. Emit structured JSONL diagnostics beside TensorBoard scalars.
7. Keep the implementation under `areno/experimental/care/` until it survives
   P2/P3.

No CUDA kernel or dependency change is required.

## Implemented hook result schema

```python
{
    "trajectories": [
        {
            "trajectory_id": "p0:s0:r0",
            "turns": [
                {
                    "turn_index": 0,
                    "weight": -0.75,
                    "sign": -1,
                    "abstain": False,
                    "confidence": 0.9,
                    "diagnostics": {},
                }
            ],
            "selected_tokens": 24,
            "budget_tokens": 64,
            "audit_calls": 1,
            "diagnostics": {},
        }
    ]
}
```

`sign` must match `weight`; `abstain` is true exactly for zero weight.
Declared selected mass is recomputed from the preserved static eligibility
mask. Update rows require a positive fixed budget and cannot exceed it. A row
may declare budget zero only when it is fully abstaining; P3 uses this narrow
case for calibration-only rows that must not reach the optimizer.

The reference hook at
`examples/agentic/trainable_turns_ablation/outcome_broadcast_turn_credit.py`
is an engineering control, not the CARe research method.

## Required CPU tests

- span offsets survive multi-call concatenation exactly;
- positive, negative, and zero turn weights map to the intended tokens;
- full abstention produces no backend/optimizer call;
- the no-backfill budget rule is nested and never exceeds `B`;
- two selections with different coverage but the same declared budget have
  the same normalization denominator;
- malformed hook output is rejected before training;
- diagnostics contain seed, step, trajectory, turn, sign, confidence, selected
  mass, masked mass, audit calls, and schema version;
- default `None` hook is byte-for-byte behavior-compatible at the constructed
  `TrainSequence` level.

All items above are covered by `tests/test_care_turn_credit_cpu.py`, including
an additional test for exact normalization across multiple backend
microbatches.

## Why a public decision is unavoidable

Keeping the hook entirely in an example would have required reaching into private
`_AgentSample` state or duplicating the trainer loop. That would create a
paper-only fork whose semantics differ from the tested AReno path. The narrow
config/CLI hook is the smallest reproducible interface that keeps the
experimental method outside the core algorithm factory while using the real
agentic batch path.

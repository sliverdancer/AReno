# Structured Action-Span Supervision Instrument

This directory is the Q0 research instrument for protocol `SAS-P0-v1.0`.
It is separate from the general shopping example because research trajectories
must never repair or synthesize model tool calls.

Properties:

- four fixed tool-action turns;
- model-generated arguments retained verbatim;
- exactly one expected call per turn;
- stable invalid-call reason codes;
- immediate stop on invalid output;
- strict four-call outcome reward;
- 80 deterministic, constraint-signature-disjoint tasks split into
  48 train, 16 qualification, and 16 held-out records.

Generate the frozen task files:

```bash
python3 examples/agentic/structured_action_supervision/dataset_generator.py \
  --output-dir research/structured_action_supervision_v1/stages/Q0/dataset
```

This is CPU preparation only. It does not authorize model downloads, serving,
or training. The held-out task file may be hashed and schema-validated before
Q3, but its model reward outcomes must not be consumed.

After Q0 passes, prepare the bounded two-arm/two-seed Q1 manifest:

```bash
python3 examples/agentic/structured_action_supervision/run_q1_pilot.py \
  --prepare
```

The prepared commands include explicit `--seed` values and use only the
qualification split. `--execute-gpu-pilot` is intentionally separate and must
not be used without explicit GPU/model authorization.


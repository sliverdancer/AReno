# RIST v3.1 terminal failure taxonomy for v4 design

This file summarizes v3.1 only as diagnosis. It is not a v4 task-selection
oracle and does not authorize qualification, held-out, BFCL, model access, or
training.

## Observed calibration failure mode

- Qwen3: 1024/1024 trajectories failed; completed_turns = 0 for all rows.
  - `WRONG_CODE`: 1019
  - `WRONG_TOOL`: 5
- Gemma4: 1024/1024 trajectories failed; completed_turns = 0 for all rows.
  - `WRONG_CODE`: 992
  - `WRONG_TOOL`: 32

## Interpretation

The v3 task contract asked models to copy 10-character hash-derived codes. That
made even the easiest cells collapse before turn 2, eliminating mixed or high
reward-resolution groups. v4 therefore removes hash-derived codes and uses short
semantic codes while keeping strict argument equality.

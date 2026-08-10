# RIST v4 strict-independent research protocol

Protocol: `RIST-C0-v4.0`

Status: `CPU_POOL_FREEZE_PASS_GPU_CLOSED`

## Scientific purpose

v4 tests whether a semantic-code synthetic transport can produce reward
resolution contrast before any supervision-topology training. The target claim
remains gradient resolvability under group-relative objectives, not novelty of
turn credit assignment itself.

## Retired parent

`RIST C0 v3.1` is terminal. It completed two-family calibration but all shared
cells collapsed to low resolution because both models failed at turn 1, mostly
with `WRONG_CODE`. v4 may use that as failure-mode diagnosis only. It must not
use v3 outcome values to select individual task rows or thresholds.

## v4 task pool

- Splits: calibration, qualification, capacity_canary.
- Calibration and qualification: 32 tasks each, 8 cells x 4 replicates.
- Rollout seeds: 32 per task for calibration and qualification.
- Each task has four ordered tool-call turns and a unique strict oracle.
- Argument codes are short semantic tokens such as `A1`, not hash substrings.
- Difficulty varies by tool decoys, argument decoys, dependency depth,
  relational selector, and semantic ambiguity.

## Access boundaries

Calibration cannot read qualification content while building its stage manifest.
Qualification remains sealed until calibration returns a frozen admission with
at least two common low cells and two common high cells across Qwen3 and Gemma4.
There is no retry, task replacement, threshold tuning, held-out/BFCL access, or
training in C0.

## Deployment binding

v4 preserves the v3.1 non-circular binding rule: target-host manifests bind only
static runtime identity. Live collector identity must separately include the
concrete deployment receipt SHA-256 after receipt generation.

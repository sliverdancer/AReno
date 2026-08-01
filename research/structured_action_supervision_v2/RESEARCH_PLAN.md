# SAS v2 Tool-Readiness Bridge — Frozen Research Plan

Plan ID: `SAS-TR-v2.0`

Date: `2026-07-30`

Status: `B2_INVALID_INFRASTRUCTURE_KV_CACHE_OOM`

## 1. Research judgment

The v1.1 outcome does not justify another training run with a changed flag. It
does justify a separate causal instrument study because all 128 responses:

- entered `<think>`;
- used the full 128-token supervised response budget;
- never closed `</think>`;
- contained no raw tool-call markup; and
- produced no parsed tool call.

The stored OpenAI response did not retain `finish_reason`. Therefore
"thinking-mode budget exhaustion caused the failure" remains a hypothesis, not
a confirmed root cause.

The current main-conference route remains killed. This bridge can reopen a
diagnostic AF/LF pilot only after it independently establishes a non-degenerate
tool-use interface.

## 2. Competing hypotheses

### H1 — Thinking-budget exhaustion

Default Qwen3 thinking consumes the short generation budget before the forced
tool call is emitted.

Prediction: `D128` has low first-call executability; either `D512` or a
non-thinking cell materially improves it.

Falsifier: `D512`, `N128`, and `N512` all remain below the frozen executable
gate.

### H2 — Thinking/action entanglement

Even if a larger budget lets default thinking reach a call, including reasoning
tokens in an `assistant_tool_call` span changes the scientific treatment from
action-span supervision to reasoning-plus-action supervision.

Prediction: only non-thinking cells are eligible for the future SAS
instrument. Default-thinking cells remain diagnostic regardless of success.

Falsifier: none inside B2; this is a construct-validity restriction derived
from the declared estimand.

### H3 — Parser/interface mismatch

The model may emit tool markup that AReno fails to parse.

Prediction: raw tool markup without a parsed call would support this mechanism.

Current evidence: v1.1 has zero raw tool markers in 128 responses, so H3 does
not explain that consumed attempt. B2 still records raw and parsed forms to
detect a new mismatch.

### H4 — Base-model capability floor

Qwen3-0.6B may be unable to complete this four-turn task even under a correct
interface.

Prediction: both `N128` and `N512` fail four-turn completion or produce
degenerate rewards.

Decision: `KILL_QWEN3_0_6B_INSTRUMENT`. Larger checkpoints are outside this
protocol and require a new model-selection rule and authorization.

## 3. B2 factorial

All cells use the same Qwen3-0.6B weight hash, task rows, tool schemas,
forced-tool sequence, temperature/top-p policy, and sampling seeds.

| Cell | Thinking | Maximum response tokens | Scientific role |
| --- | --- | ---: | --- |
| `D128` | tokenizer default | 128 | diagnose v1.1 condition |
| `D512` | tokenizer default | 512 | isolate budget relief |
| `N128` | disabled | 128 | eligible minimal interface |
| `N512` | disabled | 512 | eligible fallback interface |

Calibration uses 16 task rows crossed with seeds 3101 and 3202: 32 trajectories
per cell, 128 total. The split was selected by a salted hash before B2.

## 4. Outcomes and gates

Primary qualification outcomes:

1. first-turn exact executable call rate;
2. complete ordered four-turn protocol rate;
3. positive strict-reward rate;
4. raw/parsed tool-call agreement;
5. generation length, latency, invalid reasons, and finish reason.

A non-thinking cell passes only if:

- first-turn executable rate is at least 0.95;
- four-turn completion rate is at least 0.75;
- positive strict-reward rate is between 0.05 and 0.95;
- fabricated-call count is zero; and
- every attempted trajectory and raw response is preserved.

Selection is mechanical:

1. select `N128` if it passes;
2. otherwise select `N512` if it passes;
3. otherwise kill the Qwen3-0.6B instrument.

Default-thinking cells cannot be selected. If a cell is selected, run it once
on the untouched 16-row validation split with the same two seeds and gates.
Reserve remains unopened.

## 5. Stage gates

### B0 — Retrospective forensics

Status: `PASS_ROOT_CAUSE_CANDIDATE_THINK_BUDGET_EXHAUSTION`

This is a mechanism candidate, not causal proof.

### B1 — CPU freeze

Status: `PASS_B1_CPU_FREEZE_TO_GPU_AUTHORIZATION_REQUEST`

Required artifacts:

- deterministic calibration/validation/reserve split;
- fixed four-cell design and seeds;
- exact selection and kill rules;
- offline forensic and split tests;
- no public API, config, CLI, or dependency changes.

### B2 — GPU inference qualification

Status: `INVALID_INFRASTRUCTURE_KV_CACHE_OOM`

B2 was authorized and opened on 2026-08-01. The exact source and model hashes
passed, but the server configured eight running prompts and allocated about
22.5 GiB of KV cache. The first inference request then failed while attempting
another 642 MiB allocation with only 557 MiB free. No raw model response was
returned. The 64 HTTP failures from D128/D512 are infrastructure failures and
must not enter the scientific cell analysis. N128, N512, validation, and reserve
remain unopened. `SAS-TR-v2.0` is terminal; see `stages/B2/REPORT.md`.

### B3 — New AF/LF pilot

Status: `UNOPENED`

B3 may open only after B2 validation passes. It requires a new protocol
version, fresh manifest, separate GPU-training authorization, distinct emitted
masks on real multi-turn trajectories, and non-degenerate reward support.

## 6. Main-conference hook

B2 cannot return `GO_MAIN_TRACK`.

After B2:

- if no eligible interface passes: `KILL_CURRENT_RESEARCH_INSTRUMENT`;
- if one passes: `STAY_DIAGNOSTIC`, with B3 eligible for planning;
- a measurement-paper pivot about reasoning-budget confounding remains
  exploratory unless the interaction replicates across at least three
  checkpoints, two model families, two inference runtimes, and multiple
  executable tool benchmarks;
- the original SAS main-track route can be reconsidered only after a valid,
  powered, multi-environment AF/LF study plus a benchmark contribution. Static
  masking itself is not a novel method claim.

## 7. Bias controls

| Threat | Control |
| --- | --- |
| Post-failure tuning | New protocol, training-only source rows, salted split |
| Outcome-driven cell choice | Selection order and thresholds frozen before B2 |
| Held-out leakage | Original held-out and B1 reserve remain unopened |
| Parser repair | Preserve raw output; never synthesize calls |
| Thinking/action confound | Default-thinking cells diagnostic only |
| Model escalation | Kill at 0.6B; larger models require a new protocol |
| Main-track overclaim | B2 is qualification evidence only |

## 8. Resource ceiling

B2 is bounded to 160 trajectories maximum: 128 calibration trajectories plus
32 validation trajectories only if a cell passes. It uses one already
downloaded 0.6B checkpoint and no optimizer state. The execution controller
must stop on non-finite output, repeated server failure, evidence loss, or the
user's GPU ceiling.

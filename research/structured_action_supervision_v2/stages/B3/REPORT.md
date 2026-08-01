# B3-A Within-Group Signal Qualification

Decision: `KILL_CURRENT_GSPO_PILOT_NO_WITHIN_GROUP_SIGNAL`

The N128 interface remained perfectly executable, but the frozen GSPO signal
gate failed. Across 16 tasks and eight samples per task, every task retained
the same strict-reward outcome in all eight samples. Consequently, the current
group-relative objective would produce no non-zero advantage trajectory.

## Frozen result

| Check | Observed | Gate | Result |
| --- | ---: | ---: | --- |
| First-turn executable rate | 1.000 | at least 0.950 | pass |
| Four-turn completion rate | 1.000 | at least 0.750 | pass |
| Positive strict-reward rate | 0.250 | 0.05 to 0.95 | pass |
| Mixed strict-success groups | 0/16 | at least 4/16 | **fail** |
| Non-zero-advantage trajectories | 0/128 | at least 32/128 | **fail** |
| Fabricated calls | 0 | 0 | pass |
| Raw evidence complete | yes | yes | pass |

The aggregate 25% success rate came from four task groups that were always
successful and twelve groups that were always unsuccessful. Aggregate reward
support therefore concealed complete within-group collapse.

## Execution and containment

- exact source commit: `57ba60c`;
- model: Qwen3-0.6B, unchanged B2.1 weight hash;
- GPU: RTX 4090D;
- GPU wall time: 238.322 seconds of the 1,800-second ceiling;
- remote CPU contracts: 17 passed;
- trajectories: 128/128, with complete raw responses;
- optimizer steps and model updates: zero;
- B1 validation, B1 reserve, and original Q0 held-out: unopened by B3;
- GPU after cleanup: 0 MiB used, 0% utilization.

The downloaded evidence archive SHA-256 is
`b29820e4ff2f3f289a15763eef44a0009dc1519941cc4904e6fe2a02d6ae9ca5`.
An independent local rerun of the frozen analyzer produced the same
`signal_analysis.json` hash.

## Scientific interpretation

This is a failure of the current task/sampling/checkpoint/GSPO identification
strategy, not evidence that temporal supervision breadth or call-content
retention have no effect. AF, LF, AN, and LN training never opened, so no
treatment contrast exists.

Changing temperature, reward shaping, task selection, algorithm, or model to
create variance would define a new protocol. It must not be spliced into this
terminal result.

## Main-conference hook

Decision: `STAY_DIAGNOSTIC`; protocol action: `KILL_CURRENT_PROTOCOL`.

B3-A cannot upgrade the main-conference route because it estimates no treatment
effect. The most defensible future route, if separately authorized and
re-justified, is a new benchmark/measurement protocol that first establishes
learnable within-task variation without outcome-driven tuning.

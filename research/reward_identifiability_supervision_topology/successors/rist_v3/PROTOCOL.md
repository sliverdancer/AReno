# RIST v3 strict-independent research protocol

Protocol: `RIST-v3.0`

Status: `CPU_PROTOCOL_FOUNDATION_PASS_GPU_CLOSED`

## Scientific question

Under group-relative objectives, does base-policy conditional strict-reward
resolution moderate the effect of crossing:

- temporal action eligibility: all action turns versus final action turn; and
- structured call eligibility: complete call versus tool-name tokens only?

The four arms remain `AF`, `LF`, `AN`, and `LN`. The confirmatory interaction is
`(AN - AF) - (LN - LF)`. The claim is gradient resolvability under the chosen
estimator, not formal identifiability and not novelty of masking itself.

## Independent lineage

v2.3 ended before server launch because its orchestration hard-coded an absent
`python3`. Its calibration protocol is terminal and may not be repaired or
rerun. v3 uses new protocol identifiers, task signatures, rollout seeds,
deployment artifacts, ledgers, and output roots. No v2 outcome or cell decision
may be read by a v3 scientific process.

The sole deployment gate binds:

1. one shared clean source commit;
2. embedded manifest SHA-256;
3. model revision;
4. GPU UUID;
5. compiled extension SHA-256;
6. Python executable real path;
7. Python executable SHA-256;
8. exact Python version.

The actual child command must begin with that bound executable. `python`,
`python3`, PATH lookup, shell expansion, and fallback discovery are forbidden.

## C0 transport gate

Calibration and qualification each contain 32 tasks per family and 32 rollout
seeds per task: 1,024 trajectories per family and 2,048 per stage. Requests run
with concurrency eight and zero retry. Qualification remains inaccessible until
calibration freezes a common whole-cell map and returns the exact admission
`PASS_CALIBRATION_TO_QUALIFICATION`.

The analysis keeps the v2.3 thresholds because they were set before any v3
outcome and are not themselves the failed mechanism:

- sort 32 seeds and form four groups of eight per task;
- mixed group: contains both strict rewards;
- collapsed cell: Wilson 95% upper bound <= 0.25;
- resolved cell: Wilson 95% lower bound >= 0.50;
- at least three of four tasks must agree;
- both families need at least two common low and two common high cells in both
  calibration and qualification.

Any post-access infrastructure failure consumes the stage. There is no retry,
repair, task replacement, selective rerun, or result-conditioned threshold
change.

## Later gates

Only after C0 transports may E1 test one optimizer step for Qwen and Gemma under
GSPO and GRPO. P3 remains the paired-seed factorial with update-count and
trainable-token matching. PACT, TurnSight, SERL-SQL, TACO, CIGPO, and Signal
Reshaping are mandatory baseline or threat families before a main-conference
claim.

## Current authorization

CPU-only pool construction, access-boundary tests, exact deployment replay, and
independent audit are open. GPU use, model access, serving, inference, training,
qualification, held-out evaluation, and BFCL content remain closed.

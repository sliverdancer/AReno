# B3 Within-Group Signal Qualification

Protocol: `SAS-B3-v3.0`

Status: `FROZEN_BEFORE_GPU_EXECUTION`

## Purpose

B2.1 qualified a non-thinking 128-token tool interface, but its aggregate
reward support does not prove that GSPO receives non-zero group-relative
advantages. This stage therefore qualifies the learning signal before any
weight update. It is not an AF/LF efficacy experiment.

## Frozen preflight

- model: Qwen3-0.6B with the B2.1 weight hash;
- interface: N128, thinking disabled;
- data: the 16 already-consumed B1 calibration rows;
- sampling: temperature 1.0, top-p 1.0, top-k -1;
- eight frozen base seeds per row, producing 128 trajectories;
- one server running prompt and one client worker;
- exact four-call validator and strict reward from B2.1;
- no synthesized, repaired, filtered, or selectively repeated call.

B1 validation, B1 reserve, and the original Q0 held-out split remain unopened
by B3.

## Mechanical gate

The preflight passes only if all of the following hold:

1. first-turn executability is at least 0.95;
2. four-turn completion is at least 0.75;
3. overall positive strict-reward rate is in `[0.05, 0.95]`;
4. at least 4 of 16 task groups contain both a positive and a non-positive
   strict reward;
5. at least 32 of 128 trajectories have non-zero group-normalized advantage;
6. fabricated-call count is zero and every raw response is preserved.

Pass decision: `PASS_B3A_WITHIN_GROUP_SIGNAL_OPEN_FACTORIAL_TRAINING`.

Failure decision: `KILL_CURRENT_GSPO_PILOT_NO_WITHIN_GROUP_SIGNAL`.

A failure does not show that AF/LF or full-call/name-only supervision are
equivalent. It only shows that this task, sampling policy, checkpoint, and GSPO
group construction cannot identify those effects. Changing sampling, reward,
data, or model requires a new protocol.

## Main-conference hook

Passing B3-A can only return `STAY_DIAGNOSTIC_OPEN_FACTORIAL_TRAINING`.
It cannot support an effect direction or a main-conference claim.

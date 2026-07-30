# P2 Gate Decision

Protocol: `CARE-P2-INSTRUMENT-v0.2`
Decision date: `2026-07-29`
Decision: `PASS_P2_INSTRUMENT_TO_P3_DESIGN`
GPU status: `BLOCKED_PENDING_EXPLICIT_COMMAND_AUTHORIZATION`

## Authorization consumed

The user explicitly authorized the frozen minimum public-interface change:

- `PolicyTrainerConfig.turn_credit_fn_path`;
- `PolicyTrainerConfig.turn_credit_config_path`;
- matching training CLI options;
- internal span propagation and experimental GRPO materialization.

This authorization did not include model download, remote deployment, GPU
training, or serving. None was performed.

## Evidence

- `19/19` targeted P2 CPU contract tests passed.
- `190/190` current scoped issue #199, metrics, P1/P3, CLI-config, agentic,
  trainer-API, and P2 regression tests passed before commit.
- `git diff --check` passed.
- A real CPU `--prepare` smoke generated all three seeded arm commands, an
  eight-row dataset, and a manifest.
- Smoke dataset SHA-256:
  `5235c7d80c10c39a041a28179219dec74c46828171ddea8d7166758da3c6c8fe`.
- Smoke manifest SHA-256:
  `9cf4c871f093eb7bb434fbf65589fd116fbb844d19cbee9c1497edbe8ae0df35`.
- The complete repository CPU suite was attempted but is not a pass claim:
  the Windows Python collector stopped at `pty -> tty -> termios`, which is
  unavailable on Windows; WSL Python has no installed `pytest`. This
  platform-level collection block is outside the relevant tested path.

## What passed

1. Ordered assistant spans survive the real agentic train-batch boundary.
2. Hook input contains stable response-relative offsets, raw span material,
   static eligibility, log-probabilities, terminal reward, and group-relative
   outcome advantage.
3. Signed positive/negative credit and explicit abstention map to the intended
   response tokens.
4. Selected mass, sign, confidence, budget, row alignment, and JSON
   serializability are validated before the backend step.
5. Full-batch abstention returns an empty train batch, so the outer trainer
   does not call the backend or optimizer.
6. Fixed-budget scaling compensates both GRPO's per-microbatch token mean and
   the engine's gradient-accumulation averaging.
7. Structured JSONL diagnostics include the frozen audit fields.
8. With both new config fields at `None`, constructed `TrainSequence` objects
   preserve the previous behavior.
9. Hook/config files load during trainer construction, before GPU
   initialization and rollout.

## Deliberate restriction

P2 supports only agentic GRPO. Current GSPO collapses each trajectory to one
sequence ratio and one sequence advantage, so it cannot preserve opposite
turn signs. GSPO, PPO, non-agentic, and config-without-hook combinations fail
closed before rollout.

## Claim boundary

This is an instrument qualification, not method evidence:

- no CARe router has been implemented or evaluated;
- the bundled outcome-broadcast hook is an engineering control only;
- no learning improvement, causal attribution quality, or novelty claim is
  supported;
- the issue #199 ten-step three-arm harness remains an engineering smoke
  ablation.

## Next gate

P3 may proceed with CPU-only protocol and artifact design. Before any rented
GPU run, freeze the actual CARe router, task suite, baselines, seeds,
replication count, primary metric, statistical test, compute cap, and exact
GO/KILL rule. Then present the exact command and rental estimate for a new
explicit user authorization.

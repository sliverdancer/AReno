# X2.1 Tau3 upstream-aligned environment qualification

Status: `PASS_X2_1_ENVIRONMENT_QUALIFICATION`

X2.0 is terminal and is not rerun. X2.1 replaces the invalid assumption that
every reference action is independently executable with an upstream-aligned
environment test. It does not inspect or use the Tau3 upstream test task split.

The qualification has two independent components, each executed twice with
outbound socket connections denied:

1. Run the frozen Tau3 v1.0.1 airline and retail tool-test files in isolated
   subprocesses. Both runs must exit zero and report the same collected and
   passed counts.
2. From fresh copies of the upstream test-fixture databases, execute one fixed
   mutating tool call per domain: airline `cancel_reservation(4WQ150)` and
   retail `cancel_pending_order(#W0000000, no longer needed)`. Preserve the
   semantic tool response (excluding only the generated timestamp) and chained
   database-state hashes in strict transcripts. Both replays must be exactly
   canonical-equivalent.

Any exception, non-zero upstream test exit, differing test counts, network
attempt, invalid transcript, or replay mismatch is terminal
`KILL_X2_1_ENVIRONMENT_QUALIFICATION`. The first execution is consumed; no
selective repair or rerun is permitted.

This stage is CPU-only environment/instrument qualification. It accesses no
BFCL task content, model, tokenizer, inference, training, or GPU.

## Terminal execution

The single frozen execution passed. Both upstream runs reported 28/28 tool
tests passed. The two clean-reset airline and retail state-mutating canaries
were canonical-equivalent across replays. The archived result SHA-256 is
`b4e0002fddf2eeb06fadd9542568fa7f0e971247e03bf429a299ebeaee66148d`.

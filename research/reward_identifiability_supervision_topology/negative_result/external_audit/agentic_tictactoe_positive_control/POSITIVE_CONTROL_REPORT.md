# Agentic Tic-Tac-Toe parseable positive control

Status: `PASS_CPU_ONLY_PARSEABLE_TOOL_CALL_POSITIVE_CONTROL`

This CPU-only artifact uses the public repo-native `examples/agentic/tictactoe`
`choose_square` tool protocol as a positive control for the reward-resolution
diagnostic. It does not use BFCL, held-out data, a model, an API, GPU, or
training.

## Result

- Groups: 4
- Rollouts: 16
- Parseable tool-call rate: 1.000
- Strict success rate: 0.500
- Mixed groups: 2
- All-pass groups: 1
- All-fail groups: 1
- Non-zero-advantage groups: 2
- High/low/mixed contrast present: True

## Boundary

This is a diagnostic positive control, not an external BFCL result. It closes
one gap in the evidence chain: the analyzer can identify reward-resolution
contrast when parseable tool calls exist. The remaining gap is a true external
public environment canary that produces parseable model tool calls.

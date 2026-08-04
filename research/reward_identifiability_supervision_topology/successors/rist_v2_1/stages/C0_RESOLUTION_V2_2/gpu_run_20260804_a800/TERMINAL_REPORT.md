# RIST C0 v2.2 A800 terminal report

Status: `TERMINAL_PROTOCOL_FAILURE`

The Qwen3-0.6B outcome-free capacity request completed all eight concurrent
responses with zero retry on the bound A800. No task outcome was inspected.
However, its pre-request runtime identity recorded source commit `2674e101`,
while the current GPU execution freeze requires serving source commit
`f00b688d`. The serving process itself ran from `f00b688d`; the mismatch arose
because the identity builder and canary harness were invoked from the detached
runtime worktree, whose superseded canary manifest still named `2674e101`,
instead of from the current control worktree.

This evidence cannot be repaired by editing the identity after execution. A
Qwen rerun would violate the frozen zero-retry, fail-closed capacity protocol.
Therefore C0 v2.2 is consumed and killed without starting the Gemma canary or
opening calibration, qualification, resolution analysis, E1 training,
held-out evaluation, or BFCL.

The server was stopped and the final GPU memory use was zero. The raw response
journal, result, runtime identities, serve log, GPU identity, and compressed
archive are retained in this directory. This terminal result is an execution
protocol failure, not evidence for or against the scientific hypothesis.

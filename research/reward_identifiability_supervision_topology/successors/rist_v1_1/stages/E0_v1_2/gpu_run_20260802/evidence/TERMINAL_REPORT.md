# RIST-E0-v1.2 terminal report

Decision: `PASS_E0_INFRASTRUCTURE_TO_P2_1_PROTOCOL_FREEZE`

Both frozen native-backend serving cells passed independently. Qwen3-0.6B and
Gemma-4 E2B each produced 8/8 raw HTTP 200 responses, 8/8 parser-valid single
offered tool calls, and 8/8 exact instructed tool/code pairs, with zero retries,
fabricated calls, HTTP 5xx responses, or client infrastructure exceptions.

The sequential serving window was 142.743 seconds against the frozen
1800-second ceiling. No training occurred, no qualification or held-out rows
were opened, no checkpoint was downloaded or replaced, and final GPU and
relevant-process listings were empty.

Both servers emitted an application-shutdown warning after all responses while
handling the requested SIGTERM because a worker had already received the same
signal. This post-request cleanup warning did not affect a response, produce a
5xx/client exception, leave a process behind, or alter the mechanical validator
decision; it is retained in the raw server logs and independent audit.

E0 is infrastructure-only and cannot update the scientific or main-conference
claim. The permitted next action is CPU freezing of P2.1; scientific serving
still requires a new authorization after that freeze.

# T0b v1.1 actual runtime response-token gate

Status: `FROZEN_CPU_READY_AWAITING_SEPARATE_GPU_AUTHORIZATION`

T0b v1.1 is a new inference-only treatment-identifiability protocol. It is not
a repair or retry of terminal v1.0. It uses eight newly generated calibration
nonces and codes that are disjoint from v1.0, four turns per nonce, and the same
immutable Qwen3-0.6B and Gemma4 E2B revisions. No v1.0 task, response, or outcome
may enter a v1.1 fixture or decision.

The serving source is frozen at commit
`b6d7bdcfb0ee7be89ec1c6e16433e33ba5546db7`. A single-choice response must
contain non-empty, actual `areno.response_tokens` emitted by the engine. Token
IDs may not be reconstructed from decoded text. The public extension retains
actual input tokens and an empty logprob list; multi-choice metadata is omitted
because its current shape would be ambiguous.

Each model must independently yield exactly 32 exact calls, eight per each of
four turns, with zero retry. Native attention, eager decode, thinking disabled,
one running prompt, temperature zero, and the locked tokenizer hashes remain
mandatory. Any HTTP failure, OOM, missing/empty token metadata, malformed call,
or incomplete coverage terminates that model cell without repair.

After collection, the production mask fixture must report 32 cases, exact
name-only masks, zero tool-name boundary mixed tokens, and eight cases per turn
for both models. Passing T0b only qualifies measurement and opens C0/E1; it is
not a supervision-effect result or a main-conference upgrade.

This CPU freeze authorizes no GPU, inference, model download or replacement,
training, held-out access, or BFCL content. A separate user authorization is
required before execution.

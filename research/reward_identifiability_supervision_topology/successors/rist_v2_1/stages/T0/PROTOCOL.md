# T0 tokenizer and treatment-identifiability gate

Status: `FROZEN_FAIL_CLOSED_AWAITING_TOKENIZER_ACCESS`

The scientific contrast is full tool call versus tool-name-only supervision.
The current public switch, `--mask-tool-call-args`, implements argument-value
masking. It leaves non-argument JSON syntax and wrapper tokens trainable, so it
must not be described as tool-name-only without checkpoint-specific evidence.

## Required fixture

For each candidate checkpoint, capture at least 32 valid four-turn tool calls
from calibration nonces using the exact serving tokenizer and response token
IDs. Each case records a disjoint partition of response-token indices into:

- `name_indices`: tokens attributable only to the tool name;
- `argument_indices`: tokens touching the argument value;
- `other_indices`: syntax, keys, whitespace, wrappers, or special tokens.
- `shared_indices`: tokenizer tokens spanning a semantic boundary.

It also records the actual loss mask emitted by the AReno training path. Any
non-empty `shared_indices` fails the tool-name-only gate rather than being
silently approximated.

## Gates

- Full-call arm: every eligible response token remains trainable.
- Argument-masked diagnostic: every name token is trainable and every argument
  token is masked. This is not sufficient for the paper treatment.
- Tool-name-only arm: exactly `name_indices` are trainable; every argument and
  other token is masked.
- Pass requires all cases and all four turns to satisfy the chosen treatment,
  with zero localization failures.

Qwen and Gemma fixtures are evaluated independently. A failure may motivate a
new public `--tool-call-supervision {full,name_only}` option, but public config
and CLI changes require explicit authorization under `AGENTS.md`.

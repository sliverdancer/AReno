# T0 tokenizer and treatment-identifiability gate

Status: `PUBLIC_TREATMENT_IMPLEMENTED_AWAITING_TOKENIZER_ACCESS`

The scientific contrast is full tool call versus tool-name-only supervision.
The legacy public switch, `--mask-tool-call-args`, implements argument-value
masking. The new public `--tool-call-supervision {full,name_only}` treatment
uses exact offset mapping and fails closed on mixed-boundary tokens. It still
must pass checkpoint-specific fixtures before use as a scientific treatment.

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

Qwen and Gemma fixtures are evaluated independently. Any localization failure
rejects that checkpoint/treatment pairing; the mask must not be approximated.

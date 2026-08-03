# T0 tokenizer and treatment-identifiability gate

Status: `PUBLIC_TREATMENT_IMPLEMENTED_AWAITING_TOKENIZER_ACCESS`

The scientific contrast is full tool call versus tool-name-only supervision.
The legacy public switch, `--mask-tool-call-args`, implements argument-value
masking. The new public `--tool-call-supervision {full,name_only}` treatment
uses exact offset mapping and fails closed on mixed-boundary tokens. It still
must pass checkpoint-specific fixtures before use as a scientific treatment.

## Required fixture

For each candidate checkpoint, first run the canonical tokenizer-only preflight.
That preflight cannot qualify T0. Qualification requires exactly 32 valid
runtime tool-call responses from calibration nonces: eight actual response-token
rows from each of turns 0, 1, 2, and 3, using the exact serving tokenizer and
the production name-only mask. Each case records a disjoint partition of
response-token indices into:

- `name_indices`: tokens attributable only to the tool name;
- `argument_indices`: tokens touching the argument value;
- `other_indices`: syntax, keys, whitespace, wrappers, or special tokens.
- `shared_indices`: tokenizer tokens spanning the tool-name boundary into an
  excluded region;
- `masked_boundary_indices`: tokens spanning only argument and syntax regions,
  both of which are excluded by the name-only treatment.

It also records the actual loss mask emitted by the AReno training path. Any
non-empty `shared_indices` fails the tool-name-only gate rather than being
silently approximated. A `masked_boundary_indices` token is admissible only
when it remains masked: mixing two excluded semantic regions does not make the
tool name inexact. T0a v1.0 classified both boundary types together; the first
real-tokenizer development preflight exposed that overconstraint before any
runtime qualification data existed. T0a v1.1 separates them without relaxing
the tool-name boundary.

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
Synthetic canonical calls, re-tokenized message fields without runtime response
IDs, or unbalanced turn coverage cannot satisfy the gate.

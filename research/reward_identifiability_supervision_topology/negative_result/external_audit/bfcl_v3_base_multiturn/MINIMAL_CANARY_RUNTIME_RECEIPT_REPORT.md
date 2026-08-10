# BFCL minimal inference canary runtime receipt

Status: `TEMPLATE_FROZEN_NOT_EXECUTABLE`

This freezes the next inference step without authorizing it. The template is a
runtime receipt for the smallest admissible model canary:

- one public task: `multi_turn_base_0`;
- one model slot: `qwen_family_primary_canary`;
- one rollout;
- zero retries;
- terminal finalizer required even on parse failure;
- full 64-task external audit remains closed.

## Frozen template

- `MINIMAL_CANARY_RUNTIME_RECEIPT_TEMPLATE.json`
- SHA-256:
  `1985ef797b7f4a92a51f665e437c67ac7f1cb894a94ba651db4d9273598c83d3`

## Execution status

Not executable yet. Before the first model request, a concrete runtime receipt
must replace all `UNBOUND` fields with exact values:

- model repo id;
- model revision;
- tokenizer revision;
- model snapshot/hash;
- serving backend;
- serving command hash;
- GPU UUID or explicit API/CPU not-applicable marker;
- Python environment hash;
- extension hashes;
- output root outside the Git source tree;
- max output;
- tool-call format;
- request timeout.

## Risk judgment

The one-task Qwen-family format canary has moderate success probability,
approximately `0.50-0.65`, if the BFCL tool schema and prompt rendering are
correct. This is not high enough to skip canaries and run the full audit.

The dominant risk is strict function-call formatting failure. A parse failure is
still useful if the terminal finalizer cleanly categorizes it, but it is not
strong external reward-resolution evidence.

## Next admissible step

If inference is later authorized, bind a concrete runtime receipt from this
template and run exactly one model request. Do not run the full 64-task audit
until the one-task canary is terminal and interpretable.

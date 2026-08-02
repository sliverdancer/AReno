# Authorized public API decision

Status: `IMPLEMENTED_AWAITING_REAL_TOKENIZER_FIXTURES`

If real tokenizer fixtures confirm that `--mask-tool-call-args` retains syntax
tokens, the minimum scientifically aligned change is a new additive option:

```text
--tool-call-supervision {full,name_only}
```

The `name_only` implementation must build a response-aligned mask from exact
token offsets, keep only tool-name tokens, compose with prior suppression, and
fail when tokenizer tokens mix tool-name characters with syntax. It must include
Qwen3 and Gemma4 real-tokenizer fixtures plus CPU fake-tokenizer regression
tests. Existing `--mask-tool-call-args` behavior remains backward compatible.

The additive option is implemented in `TrainerConfig`, `LossMaskPolicy`, and the
CLI after explicit authorization. `capture_mask_fixture.py` now calls the same
production mask implementation, requires an isolated tokenizer-only snapshot,
and rejects any model-weight file before reading it. Scientific execution
remains blocked until Qwen3 and Gemma4 fixtures verify exact offsets through
that production path.

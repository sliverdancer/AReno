# Deferred public API decision

No AReno public API was changed in T0.

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

This proposal is intentionally unimplemented because `AGENTS.md` requires an
explicit decision before changing public config dataclasses or CLI surfaces.

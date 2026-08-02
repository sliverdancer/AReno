# RIST-v2.1 D2.1 pre-freeze HTR contract

M0 commit: `49b09ec03caf1daf4714630757421ad59c89bb3f`

M0 archived score: `7/8`

Objective: raise the successor CPU split-local evaluator to 8/8 without
changing frozen RIST-v2 D2, generator semantics, data, thresholds, or accessing
held-out.

Development input:

- split: calibration;
- rows: 32;
- SHA-256: `5ab3a7245ab2f3cda8eba1f641927bdb4bb80ed2ea18db597099d8538c22c8fa`;
- reusable during search.

Test input:

- split: qualification;
- rows: 32;
- SHA-256: `07a56fc3662ff1e1260bdbf3fe32b2312f6c2a577848c48d5c47c1c000fbe82a`;
- one-shot across the entire HTR run.

The fixed harness regenerates only the requested split using the archived task
builder. It never calls the archived three-split `generate_splits` or
`write_splits`. Comparator candidates receive bytes and JSON-domain fragments,
not paths, and may not perform imports or I/O.

Eight equally weighted gates:

1. balanced eight cells;
2. unique within-split signatures;
3. unique four-turn oracles;
4. difficulty dimensions vary;
5. no model-free resolution labels;
6. byte-identical regeneration;
7. held-out inaccessible;
8. CPU-only, no model or training.

Predeclared hypotheses:

- H1: generated JSONL bytes are the reproducibility source of truth; Python
  container representation equality must not override equal bytes.
- H2: normalize manifest fragments into the JSON domain, then require both
  fragment equality and file-byte equality.

Highest dev score wins. If tied, H1 wins because it directly implements the
frozen byte gate. Test results may not guide selection or further changes.

# ARCA anonymous artifact

This CPU-only artifact supports the source-pinned reward-contract audit in the
anonymous manuscript. It does not train or serve a model and does not require a
GPU.

The manuscript uses the official TMLR style at the revision and hashes in
`research/silent_reward_contracts/p6/paper/style_manifest.json`. Place
`tmlr.sty`, `tmlr.bst`, and the style's `fancyhdr.sty` beside `main.tex` before
compiling.

## Verify an extracted archive

From the archive root:

```bash
python3 verify_artifact.py .
python3 -m pytest tests/test_arca_p5_cpu.py -q
```

The first command checks the exact allowlist, SHA-256 hashes, JSON/CSV syntax,
and anonymity patterns. The test command exercises the frozen auditor and P5
artifact generation without importing the GPU training stack. Source pins are
recorded in the P3 and P5 manifests.

## Reproduce the frozen CPU summaries

```bash
python3 -m research.silent_reward_contracts.evaluate_auditor \
  --split dev --output-dir reproduced/dev
python3 -m research.silent_reward_contracts.evaluate_auditor \
  --split heldout --output-dir reproduced/heldout
python3 -m research.silent_reward_contracts.summarize_p3 \
  --output-dir reproduced/summary
python3 -m research.silent_reward_contracts.p5.external_validation \
  --output-dir reproduced/p5
python3 -m research.silent_reward_contracts.p6.validate_paper_claims \
  --output reproduced/paper_claim_audit.json
```

These commands use the shipped, already extracted case records. Re-extracting
facts from upstream projects additionally requires checking out the exact
commits and source hashes in the manifests. No P4 scientific trajectory exists;
the redacted P4 gate record is included only to document that the attempted GPU
protocol was invalid before model loading.

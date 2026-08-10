# BFCL v3 Base Multi-Turn External Audit Feasibility

This directory contains CPU-only, outcome-blind metadata for using the public
BFCL v3 Base Multi-Turn split as an external audit target.

Do not place raw BFCL prompt, answer, or function-documentation files here.
`EXECUTION_RECEIPT_STATIC.json` freezes the external-audit execution design but
does not authorize model/API/GPU inference.

Rebuild with:

```bash
python research/reward_identifiability_supervision_topology/negative_result/build_bfcl_feasibility_audit.py
```

Set `BFCL_FEASIBILITY_SOURCE` if the public files are mirrored outside the
default temp directory.

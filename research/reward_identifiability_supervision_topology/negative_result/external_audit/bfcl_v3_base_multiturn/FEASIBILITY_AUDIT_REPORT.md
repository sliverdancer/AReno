# BFCL v3 Base Multi-Turn CPU-Only Feasibility Audit

Status: `PASS_FOR_PROTOCOL_DESIGN_ONLY`

Scope:

- Public BFCL v3 Base Multi-Turn files only.
- No model inference, no training, no GPU, no API calls.
- No sealed or held-out split access.
- Raw BFCL prompts, possible answers, and function docs are not copied into this repository.

Key source facts:

- Task records: 200
- Unique task ids: 200
- Possible-answer records: 200
- Function-doc files: 8
- JSON format: whitespace-separated JSON object stream.
- Deterministic candidate subset: first 64 ids after natural numeric task-id sort.

Feasibility conclusion:

BFCL v3 Base Multi-Turn is feasible as an external public audit target for the
reward-resolution collapse paper. The public split provides multi-turn tasks,
oracle answers, and function documentation sufficient to define a pre-inference
audit protocol. The next step must be a separate frozen execution protocol that
binds source file hashes, the selected ids, model revisions, decoding settings,
and terminal finalization rules before any model request is sent.

Important limitation:

This audit does not establish reward-resolution evidence. It only establishes
that a public, non-held-out BFCL split can be used for an outcome-blind external
audit. Any reward-resolution result requires a newly frozen execution receipt and
separate authorization for inference.

Selection rule:

`natural_numeric_sort(task_id)[:64]`

This rule is outcome-blind: it uses task ids only and does not inspect model
responses or reward outcomes.

Artifacts:

- `PUBLIC_FILE_MANIFEST.json`: public source URLs, byte sizes, SHA-256 hashes.
- `SCHEMA_SUMMARY.json`: derived schema/statistical summary only.
- `SELECTED_TASK_IDS.txt`: deterministic public task-id subset.

# External audit protocol freeze

Protocol: `RRC-EXTERNAL-AUDIT-BFCL-V3-BASE-MULTITURN-v1`

Status: `FORMAT_LAYER_ISSUE_PROBABLE_FULL_AUDIT_CLOSED`

This protocol specifies the external evidence needed before targeting NeurIPS
Evaluations & Datasets. It does not authorize model inference, API calls, GPU
use, held-out access, or training. A CPU-only public-file feasibility audit has
been completed without model outputs; see
`external_audit/bfcl_v3_base_multiturn/`.

The static execution receipt is frozen at
`external_audit/bfcl_v3_base_multiturn/EXECUTION_RECEIPT_STATIC.json`.
The CPU-only synthetic replay is recorded at
`external_audit/bfcl_v3_base_multiturn/SYNTHETIC_REPLAY_RESULT.json`.
The minimal canary runtime receipt template is recorded at
`external_audit/bfcl_v3_base_multiturn/MINIMAL_CANARY_RUNTIME_RECEIPT_TEMPLATE.json`.
The local preflight block report is recorded at
`external_audit/bfcl_v3_base_multiturn/MINIMAL_CANARY_LOCAL_PREFLIGHT_BLOCKED.json`.
The remote one-request canary terminal finalizer is recorded at
`external_audit/bfcl_v3_base_multiturn/MINIMAL_CANARY_TERMINAL_FINALIZER.json`.
The format investigation is recorded at
`external_audit/bfcl_v3_base_multiturn/FORMAT_INVESTIGATION_REPORT.md`.

## Selected public target

Primary external audit target:

- Benchmark: Berkeley Function Calling Leaderboard V3
- Subset: Base Multi-Turn public split
- Rationale: public multi-turn/multi-step function-calling tasks with exact tool
  invocation evaluation; closest external analogue to RIST's strict tool-call
  reward-resolution setting.

Fallback if the public split cannot be accessed or audited cleanly:

- ToolSandbox public scenarios, using only public scenario definitions and
  non-held-out released evaluation artifacts if available.

## Audit question

Do public multi-turn tool-use tasks exhibit reward-resolution collapse under
grouped rollout sampling, measured before any training or topology comparison?

## Frozen sampling design

If execution is later authorized:

- Models: same two-family principle as RIST, using one Qwen-family model and one
  Gemma-family model if licenses and resources permit.
- Group size: 32 rollout seeds per task.
- Sampling: temperature 0.7, top-p 0.95, max output bounded by benchmark
  evaluator requirements.
- Retries: zero.
- Task cap for first audit: 64 public base multi-turn tasks, selected by natural
  numeric sort over task IDs after schema validation.
- No task replacement after outcomes are observed.

The frozen selected ids are stored in
`external_audit/bfcl_v3_base_multiturn/SELECTED_TASK_IDS.txt`.

## Metrics

For each task group:

- strict success count;
- all-fail indicator;
- all-pass indicator;
- mixed reward indicator;
- invalid tool-call rate if the evaluator exposes it.

For each structural bucket or benchmark category:

- mixed group count;
- all-fail collapse rate;
- all-pass collapse rate;
- non-zero-advantage group rate;
- cross-model transportability of high/low labels.

## GO/KILL rule

The external audit supports a NeurIPS E&D submission only if it yields at least
one of the following:

- collapse is observed in a non-trivial fraction of public task groups, showing
  the diagnostic is not RIST-specific; or
- the public audit does not collapse, but clearly distinguishes RIST as an
  instrument-specific failure mode and demonstrates the diagnostic's ability to
  separate usable from unusable task pools.

The audit kills the high-venue route if:

- data access requires sealed or hidden test labels;
- task selection must be adjusted after observing outcomes;
- benchmark licensing prevents reproducible anonymous review artifacts;
- results cannot be summarized without private model outputs or non-shareable
  traces.

## Access boundaries

- Do not use BFCL sealed/private test content.
- Do not use leaderboard hidden evaluation as a tuning source.
- Do not train.
- Do not compare supervision topology arms.
- Do not open any RIST qualification or held-out data.
- Do not mutate RIST v3.1/v4.0.

## CPU-only feasibility audit addendum

The feasibility audit used only public BFCL V3 Base Multi-Turn files and
produced derived metadata, not raw benchmark contents.

Observed public split facts:

- Task records: 200.
- Possible-answer records: 200.
- Function-documentation files: 8.
- JSON format: whitespace-separated JSON object stream.
- Task id range after natural numeric sort: `multi_turn_base_0` through
  `multi_turn_base_199`.
- First external-audit candidate subset: `multi_turn_base_0` through
  `multi_turn_base_63`.

The complete public-file byte sizes and SHA-256 hashes are recorded in
`external_audit/bfcl_v3_base_multiturn/PUBLIC_FILE_MANIFEST.json`.

This addendum is a protocol clarification made before any model inference,
reward computation, or outcome inspection. It does not change the scientific
rule that task selection cannot be adjusted after observing model outcomes.

## Static execution receipt

The frozen static receipt records:

- public file identity;
- selected task-id file hash;
- group size, sampling parameters, request order, zero-retry policy;
- finalization and GO/KILL rules;
- explicit denial of model/API/GPU/training authorization;
- pre-inference gates required before any model spending.

Receipt hash:

`6e780bfba8d61a3ccafa8b3510cfc54894855e84cc2eefa39cc24b9a205668ca`

The receipt is non-circular: `EXECUTION_RECEIPT_STATIC.json` is hashed by
`EXECUTION_RECEIPT_STATIC.sha256`, and the containing Git commit binds both.

## CPU-only synthetic replay

The synthetic replay verifies evaluator/finalizer wiring without BFCL raw
content or model outputs. It covers exact pass, wrong tool, wrong argument,
empty observed calls, normalized JSON-string arguments, and all finalizer
branches: mixed, all-pass, and all-fail.

This replay is not external empirical evidence. It only proves that the
pre-inference gate can run and fail fast before any model request.

## Minimal canary runtime receipt template

The next inference step is constrained to one public task, one model slot, one
rollout, and zero retries. The template is not executable while any runtime
field remains `UNBOUND`.

Template hash:

`1985ef797b7f4a92a51f665e437c67ac7f1cb894a94ba651db4d9273598c83d3`

The full external audit remains unauthorized until a terminal one-task canary is
complete and interpretable.

The first local preflight did not execute because runtime bindings were missing:
no local Qwen/Gemma snapshot, CPU-only PyTorch, no vLLM/accelerate runtime, and
no concrete remote GPU/API target.

## Minimal canary terminal result

The remote A800 canary executed exactly one Qwen3-0.6B request against
`multi_turn_base_0` first turn. It terminated as interpretable `PARSE_FAILURE`:

- model requests: 1;
- retries: 0;
- expected tool calls: 3;
- observed parseable tool calls: 0;
- strict success: false;
- go to two-model canary: false;
- full BFCL audit authorized: false.

This result blocks full-audit execution. The next admissible work is
format/prompt investigation without outcome-conditioned task selection, followed
by a newly frozen one-request canary if the interface is changed.

## Format investigation result

The follow-up CPU-only investigation finds a probable format-layer issue:

- BFCL tool schemas use `parameters.type = "dict"` rather than JSON Schema
  `type = "object"`;
- direct `transformers.generate` does not enforce tool-choice output;
- the terminal canary parsed zero calls, so reward-resolution evidence was never
  reached;
- the selected first turn expects three calls, which is too strict for the first
  interface canary.

No additional rollout sampling is authorized by this result.

## Deliverables when executed

- `EXTERNAL_AUDIT_MANIFEST.json`
- `EXTERNAL_AUDIT_EVIDENCE_SUMMARY.csv`
- `EXTERNAL_AUDIT_REPORT.md`
- evaluator version and dataset commit/revision hashes;
- raw-response hash manifest, if model inference is performed.

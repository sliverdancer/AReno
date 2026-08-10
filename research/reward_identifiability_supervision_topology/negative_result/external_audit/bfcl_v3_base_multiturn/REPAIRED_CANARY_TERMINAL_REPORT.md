# BFCL repaired-format canary terminal report

Status: `TERMINAL_INTERPRETABLE_PARSE_FAILURE`

The repaired-format one-request canary was executed after applying the CPU-only
BFCL schema adapter.

Execution scope:

- task: `multi_turn_base_0`, first turn only;
- model: `Qwen/Qwen3-0.6B`;
- revision: `c1899de289a04d12100db370d81485cdf75e47ca`;
- backend: direct `transformers.generate`;
- format repair: BFCL `parameters.type = dict` converted to JSON Schema
  `type = object`;
- converted tool schemas: 32 / 32;
- model requests: 1;
- retries: 0;
- training: no;
- held-out/sealed access: no;
- full BFCL external audit: not authorized.

Terminal result:

- status: `TERMINAL_INTERPRETABLE`;
- success gate: parseable tool-call emission;
- success gate passed: false;
- strict success: false;
- expected call count: 3;
- observed parseable tool calls: 0;
- failure mode: `PARSE_FAILURE`;
- go to two-model canary: false.

The raw model response is not committed. It remains represented only by
`raw_response_sha256`.

Interpretation:

The schema adapter fixed a plausible prompt-schema issue but did not make
Qwen3-0.6B emit parseable tool calls under direct `transformers.generate`.
Additional rollout sampling is not justified. Full BFCL audit and two-model
canary should remain closed.

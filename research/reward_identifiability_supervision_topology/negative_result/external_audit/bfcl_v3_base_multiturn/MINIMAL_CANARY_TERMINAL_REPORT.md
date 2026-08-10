# BFCL minimal inference canary terminal report

Status: `TERMINAL_INTERPRETABLE_PARSE_FAILURE`

The authorized one-task, one-model, one-rollout canary was executed on the
remote A800 instance after repairing cross-platform LF receipt hashes.

Execution scope:

- task: `multi_turn_base_0`, first turn only;
- model: `Qwen/Qwen3-0.6B`;
- revision: `c1899de289a04d12100db370d81485cdf75e47ca`;
- backend: direct `transformers.generate`;
- model requests: 1;
- retries: 0;
- training: no;
- held-out/sealed access: no;
- full BFCL external audit: not authorized.

Terminal result:

- status: `TERMINAL_INTERPRETABLE`;
- strict success: false;
- expected call count: 3;
- observed parseable tool calls: 0;
- failure mode: `PARSE_FAILURE`;
- go to two-model canary: false.

The raw model response is not committed. It remains represented only by
`raw_response_sha256` in `MINIMAL_CANARY_TERMINAL_FINALIZER.json` and
`MINIMAL_CANARY_REMOTE_HASH_MANIFEST.json`.

Interpretation:

This canary does not support opening the full 64-task BFCL audit. The next
scientifically valid step is not more rollout sampling; it is a prompt/tool
format investigation using synthetic or non-outcome-bearing checks, followed by
a newly frozen one-request canary if the format issue is corrected.

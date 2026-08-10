# Tau3 Qwen3-0.6B single-request parseability canary terminal report

Status: `TERMINAL_PARSE_FAILURE`

This was the authorized Tau3/Tau2 airline parseability canary:

- execution unit: `1 public airline task x 1 model x 1 rollout`;
- model: `Qwen/Qwen3-0.6B`;
- model revision: `c1899de289a04d12100db370d81485cdf75e47ca`;
- GPU: `GPU-b599186b-7114-be7b-36dc-74b07d12dac8`;
- request backend: one local `transformers.generate` call;
- retry count: `0`;
- BFCL used: `false`;
- held-out/sealed data accessed: `false`;
- training used: `false`;
- raw response text committed: `false`.

## Terminal result

The finalizer recorded:

- model requests sent: `1`;
- observed parseable tool calls: `0`;
- parseable tool calls: `false`;
- success gate passed: `false`;
- failure mode: `PARSE_FAILURE`;
- reward-resolution claim allowed: `false`;
- go to reward-resolution calibration: `false`.

This terminal result does not satisfy the project goal of finding an external
public environment/model route that emits parseable tool calls. It closes the
current Tau3/Qwen3-0.6B single-request canary route.

## Artifact hashes

- `TAU3_RUNTIME_RECEIPT_BOUND.json`:
  `1415fdab23668857f82d254c49075caaf4846b35b2fde8701e7c1d3419391e76`
- `TAU3_RUNTIME_RECEIPT_BOUND.json.sha256`:
  `9796f38b563dd44c7c366fc42986c0381625c1365fdc5dd80886212f8748f0f7`
- `TAU3_REQUEST_PLAN_DRY_RUN.json`:
  `9f5ff9f0b03f9dccbe8f4fdc3b3386ae7af0e6e69d26596c2a6b5f97bdaccddf`
- `TAU3_CANARY_OBSERVATION.json`:
  `c4b10c653c716ff1ae70d6dc43157a5c43bdce662a4ff79dbb1e9d003a8ec1fa`
- `TAU3_CANARY_TERMINAL_FINALIZER.json`:
  `25410cafe00b3a8360bbbde8e93775c09a0f6ebf756536bc00af53aa69d489eb`

## Source identity note

The remote repository checkout was `68bbd82462dd4b2418fe0199bec802df61dd35da`.
The runner file used for the final successful execution path included the local
transformers fixes from the later local working state and had SHA-256:

`a0b60d4fa75d30f3d9c3139d358884b4bade3c4ae9dbc0ff0f9ef0e3c276df39`

The finalizer source commit field remains the bound receipt commit
`68bbd82462dd4b2418fe0199bec802df61dd35da`; this report binds the runner file
hash explicitly because the remote Git fetch of the later commit timed out.


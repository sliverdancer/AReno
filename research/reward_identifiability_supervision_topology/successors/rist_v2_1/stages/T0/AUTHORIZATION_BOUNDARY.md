# T0 authorization boundary

Status: `AWAITING_TOKENIZER_ONLY_ACCESS`

The next executable action is T0a, not inference or training. It requires a
new explicit authorization to resolve immutable revisions and download only
tokenizer/configuration files for:

- `Qwen/Qwen3-0.6B`
- `google/gemma-4-E2B-it`

The download must use isolated directories and an allowlist matching
`tokenizer_snapshot_manifest`. Files with model-weight suffixes, unexpected
files, and symlinks are rejected before content loading. T0a may instantiate
`AutoTokenizer` with `local_files_only=True`, `trust_remote_code=False`, and a
fast tokenizer. It may run the 32-case canonical preflight. It may not load a
model, infer, train, serve, use a GPU, access BFCL content, or claim T0 passed.

T0b is a later independent model-serving authorization. It must collect a
fresh calibration journal containing actual AReno `response_tokens`, then
select exactly eight valid responses from each of four turns. Only T0b can
produce a qualifying runtime fixture. A canonical tokenizer-only preflight is
necessary but cannot substitute for runtime response-token evidence.

No T0a or T0b action has been executed under the current authorization.

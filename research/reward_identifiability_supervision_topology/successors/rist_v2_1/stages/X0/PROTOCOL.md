# RIST-v2.1 X0 external-environment contract

Status: `CPU_MOCK_ONLY_AWAITING_DOWNLOAD_AUTHORIZATION`

## Three-layer evidence stack

1. RIST-v2.1 synthetic anchor for the causal factorial.
2. Tau3 text airline/retail for stateful real-tool training replication.
3. BFCL non-live `multi_turn_*` as sealed external transfer evaluation only.

The legacy tau-bench tasks are excluded because upstream marks them outdated.
BFCL official evaluation tasks must never be used for training or hyperparameter
selection. Tau3 and BFCL must run in isolated environments rather than become
AReno runtime dependencies.

## Current boundary

Neither environment is installed locally. This stage validates only a generic
reset/step/state-hash/raw-call transcript contract using synthetic fixtures.
It cannot claim upstream compatibility, license completeness, deterministic
gold replay, or real-environment qualification.

Downloading exact upstream commits, installing isolated dependencies, and
reading development metadata require separate authorization. GPU/model use and
sealed confirmatory content remain separately closed.

## Future strict gates

- exact upstream tag/commit, license, notices, lockfile, and artifact hashes;
- reset plus gold-action replay is deterministic;
- primary outcome is strict state/communication or AST/execution scoring, not
  an LLM judge;
- train/development/confirmatory IDs are disjoint;
- raw tool calls, results, state hashes, rewards, and failures are complete;
- live, web, voice, knowledge, and network-dependent categories are excluded;
- BFCL non-live evaluation remains sealed and is consumed only after training.

Primary upstream references:

- Tau3/Tau2 repository: https://github.com/sierra-research/tau2-bench
- BFCL/Gorilla repository: https://github.com/ShishirPatil/gorilla

# P2 cross-system CPU audit

Decision: `PASS_P2_CROSS_SYSTEM_TO_P3`

## Frozen result

| Framework | Natural class | Severity | Evidence |
|---|---|---|---|
| AReno | serialized argument type mismatch | high | production normalizer + production reward, dynamic CPU |
| veRL | `tool_rewards` / `reward_scores` namespace mismatch | high | pinned source dataflow + production manager, dynamic CPU |
| veRL | timeout / legitimate-zero status conflation | medium | production manager exception path, dynamic CPU |

The veRL probe loads the exact pinned `NaiveRewardManager.__call__` source. It
stubs only Ray/DataProto registration infrastructure so that the reward-boundary
code can execute in the local CPU environment without installing or serving
veRL. This is stronger than a text search but weaker than end-to-end veRL
training; the claim is therefore limited to the audited manager contract.

## Key observations

- Function tools document and emit `(response, reward)` values.
- `ToolAgentLoop` stores them under `tool_rewards`, and the agent-loop serializer
  preserves that key.
- The default reward manager provides `compute_score` with
  `rollout_reward_scores` sourced from a different key, `reward_scores`.
- In the dynamic probe, `tool_rewards=[1.0]` is absent from `compute_score`;
  observed score is `0.0` and the registered contract oracle is `1.0`.
- A raised `TimeoutError` continues with numeric reward `0.0`. The console line
  survives, but the returned structured reward artifact has no timeout status.

Commit `bf7aac2fa77fb4a7df2a931c576292d11fcbff72` is retained as intent evidence:
its title and message state that rollout/tool rewards should be exported to
`compute_score`. The audit does not infer maintainer intent only from naming.

## Claim boundary

No GPU, model, Ray cluster, or veRL trainer was started. These results establish
three source-reachable CPU contract failures across two pinned systems. They do
not estimate field prevalence, downstream model-quality loss, or whether a
particular veRL recipe separately compensates in a custom reward function.

## Artifact hashes

- CSV: `4c06471526d7829461ff0817808df4b8ed952e342852975732bb8bc37cf33e29`
- JSON: `430ccc5104ae4b59e522266bc983d58cfabeaf6d40b3a3a4b46b53f997aeeb40`

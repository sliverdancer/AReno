# X3 Tau3 stateful training adapter

Status: `CPU_ADAPTER_BUILD_AWAITING_USER_SIMULATOR_AND_MODEL_AUTHORIZATION`

X3 selects only the 88 Tau3 IDs frozen as `training` in X2. The upstream loader
stores train and retired-test tasks in one source file and may parse that file
to resolve an allowed ID; only allowlisted training task objects may enter an
episode or artifact. Retired-test tasks are never copied, prompted, scored,
selected, or used as a confirmation set. Airline and retail are preserved as
separate domains in every result.

Each policy rollout runs a fresh Tau3 `AgentGymEnv` with its real database,
tools, policy, task-specific user scenario, and upstream `EvaluationType.ALL`
binary reward. Policy calls use the AReno rollout proxy with zero retry. The
Tau3 user simulator is a distinct nuisance component and must be held fixed
across all policy families, algorithms, arms, and seeds:

- immutable user-simulator model revision;
- temperature zero;
- a seed derived before outcomes from policy seed, task ID, and sample index;
- `num_retries=0`;
- the terminal upstream `SimulationRun`, including user messages and provider
  `raw_data`, retained and hashed together with policy responses. The frozen
  task ID and source recover the corresponding user-simulator request context.

Policy malformed actions are scientific failures with reward zero. Network,
timeout, missing reward metadata, evaluator exceptions, or user-simulator
failures are infrastructure failures and invalidate the run rather than being
silently scored zero. Runtime rewards are keyed by prompt/sample and may not be
shared across the eight rollouts in a group.

CPU schema validation found that all 22 frozen airline training tasks use only
`DB` and `COMMUNICATE`, while 65 of 66 retail tasks include `NL_ASSERTION` and
would invoke an additional LLM judge under `EvaluationType.ALL`. X3 therefore
opens only the 22 airline tasks and blocks the entire retail domain. It does not
select the lone DB-only retail task because that domain-internal selection would
create an outcome-definition confound. Retail can return only under a separately
frozen non-LLM reward protocol and must be reported as an auxiliary state-only
setting, not the upstream full task reward.

X3 first runs a one-seed airline development pilot. No X3 rollout, model, user
simulator, training, or GPU action is authorized by this CPU adapter stage.

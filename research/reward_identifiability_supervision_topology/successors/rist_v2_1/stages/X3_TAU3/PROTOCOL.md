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

Every raw and reward event is additionally keyed by AReno's monotonic
`training_step`. The complete episode key is `(training_step, sample_index)`;
`prompt_index` alone is not an episode identifier because batch size is one.
The pilot result validator requires the exact 25 by 8 key grid for every cell,
reconstructs each episode's ordered raw evidence, and checks its SHA256 against
the reward journal. Exactly one terminal upstream `SimulationRun` is required
per episode, and the union of observed IDs must cover all 22 allowed airline
training tasks.

Tau3's synchronous orchestrator runs in a background thread that cannot be
killed by cancelling `asyncio.to_thread`. After normal completion, timeout, or
cancellation, the adapter therefore sends `done` only when the environment is
waiting for an agent action, waits for `_simulation_done`, and joins the frozen
upstream orchestrator. Failure to confirm termination aborts the entire process;
no later scientific episode may run in that process. A failure in one of the
eight sibling rollouts cancels and awaits the other seven before propagating.

CPU schema validation found that all 22 frozen airline training tasks use only
`DB` and `COMMUNICATE`, while 65 of 66 retail tasks include `NL_ASSERTION` and
would invoke an additional LLM judge under `EvaluationType.ALL`. X3 therefore
opens only the 22 airline tasks and blocks the entire retail domain. It does not
select the lone DB-only retail task because that domain-internal selection would
create an outcome-definition confound. Retail can return only under a separately
frozen non-LLM reward protocol and must be reported as an auxiliary state-only
setting, not the upstream full task reward.

X3 first runs a 25-step, one-seed airline development pilot containing every
Qwen/Gemma, GSPO/GRPO, and AF/LF/AN/LN cell: 16 runs and 200 fresh episodes per
run. This is an environment viability and variance instrument, not stable
interaction evidence. The user-simulator revision is deliberately unbound in
the CPU manifest; it must be fixed and independently authorized before any
episode. No X3 rollout, model, user simulator, training, or GPU action is
authorized by this CPU adapter stage.

An artifact-backed PASS additionally requires, for all 16 cells, immutable
source/model/tokenizer/GPU/user-simulator identity; raw/reward journal hashes;
metrics and checkpoint-manifest hashes; zero policy and user retries; and 3,200
validated episodes in total. Such a PASS qualifies the instrument only and is
not itself stable interaction evidence or a paper result.

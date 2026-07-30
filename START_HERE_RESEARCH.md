# AReno Scientific Research Protocol

## Document control

| Field | Value |
| --- | --- |
| Protocol version | `0.4.0-draft` |
| Repository | AReno |
| Repository version | `0.0.6` |
| Pinned starting commit | `ea95ea2fb1fc6ae290337f5e976a662b6ccf803c` |
| Status | `TRACK2_P2B_V1_VALID_NEGATIVE_PILOT_CLOSED` |
| Primary owner goal | Determine whether context-limit deletion before group-relative advantage normalization is scientifically material in AReno agentic training. |
| Current authorization | Preserve, inspect, and reviewer-audit completed Track 2 G0, P2a, and P2b-v1 artifacts |
| Not authorized yet | Any P2b rerun or redesign, further model/dataset download, paid API use, GPU training/serving, mitigation experiment, formal held-out evaluation, or external-framework claims |

This file is the authoritative starting point for research work in this
checkout. Engineering documentation remains authoritative for API and runtime
behavior, but an engineering smoke test is not scientific evidence.

## First-turn behavior

An agent or researcher starting from this document must:

1. Read `AGENTS.md`, `CODEMAP.md`, `README.md`, this protocol, and only the
   repository-local skill relevant to the next authorized stage.
2. Record:
   - `git rev-parse HEAD`;
   - `git status --short`;
   - AReno version from `pyproject.toml`;
   - available CPU, GPU, CUDA, Python, PyTorch, and disk resources;
   - the exact model, dataset, and code sources proposed for use.
3. State the current stage and its authorization before running anything.
4. Stop if the pinned protocol, dataset manifest, model source, implementation,
   or evaluation contract has changed unexpectedly.
5. Never treat model loading, a smoke test, a single training step, a reward
   curve, or a selected successful trajectory as evidence of learning quality.

## 1. Research scope

### 1.1 Primary research objective

Determine whether AReno can provide a correct, reproducible, and resource-aware
single-node execution path for LLM post-training without relying on an external
training or inference backend.

This objective is divided into separate evidence tracks:

1. **Correctness and reproducibility**
   - Do AReno's algorithms, masks, rewards, log-probabilities, gradients,
     checkpoints, and runtime boundaries behave as specified?
   - Can independent reruns reproduce the same deterministic artifacts or
     remain within preregistered numerical tolerances?
2. **Learning effectiveness**
   - On a frozen task distribution, does an AReno-trained policy improve the
     preregistered primary outcome over the base policy and matched controls?
3. **Systems efficiency**
   - Under matched model, data, algorithm, topology, sequence limits, and
     quality targets, does AReno improve a preregistered resource metric over a
     faithful public baseline without a correctness regression?

These tracks must not be merged into one claim. Correctness does not prove
learning, learning does not prove systems efficiency, and throughput does not
prove correctness.

### 1.2 Initial study boundary

The first authorized scientific track should be **correctness and
reproducibility**. It is the prerequisite for all later learning and efficiency
claims.

Agentic learning effectiveness is a follow-on track. Candidate deterministic
environments already present in the repository include:

| Candidate | Scientific use | Main risk |
| --- | --- | --- |
| Tic-Tac-Toe | Small deterministic action task with a minimax oracle | May be too easy and have limited novelty |
| Codebreaker | Partial-observability and tool-history reasoning | Sparse success and high sample demand |
| Shopping | Multi-step constraint satisfaction and tool-order compliance | Reward may mix outcome and process effects |
| DuelGrid | Stateful multi-action planning with replayable behavior | Existing before/after claims lack a frozen protocol |
| Coding | Multi-turn tool use with executable tests | Task heterogeneity and reward leakage require stronger controls |

P0 must select exactly one primary task for the first learning study. Other
tasks may be secondary replications only after the primary protocol is frozen.

### 1.3 Non-goals

This protocol does not currently authorize claims that:

- AReno is state of the art;
- AReno is faster or more memory-efficient than another framework;
- GSPO, GRPO, or PPO is superior in general;
- a reward increase implies improved general reasoning;
- a deterministic synthetic task implies real-world agent reliability;
- a smoke run or five-step CI job demonstrates convergence;
- one visually selected trajectory demonstrates systematic improvement.

## 2. Research questions and falsifiable hypotheses

### RQ1: Algorithmic correctness

Can each selected AReno algorithm reproduce its defining tensor-level
semantics on controlled inputs?

- **H1:** For a frozen set of inputs, masks, rewards or preferences,
  advantages, log-probabilities, losses, and gradients match an independent
  reference implementation within preregistered absolute and relative
  tolerances.
- **Falsifier:** Any required invariant fails outside tolerance, or the
  tolerance must be relaxed after observing the result.

### RQ2: Runtime reproducibility

Can the same frozen workload be repeated without unexplained state,
checkpoint, scheduling, or metric drift?

- **H2:** Deterministic CPU artifacts are byte-identical where exact equality
  is expected; bounded numerical GPU artifacts remain within preregistered
  tolerances and produce the same pass/fail decision.
- **Falsifier:** Reruns disagree on required artifacts or decisions without a
  documented nondeterministic source included in the protocol.

### RQ3: Agentic learning effectiveness

Does post-training improve frozen held-out task success rather than merely
increase training reward?

- **H3:** The trained policy improves held-out task success over both the base
  policy and a no-update control under matched decoding and evaluation budgets.
- **Falsifier:** The preregistered effect threshold is not met, confidence
  intervals include the no-improvement region, or gains disappear under a
  leakage-resistant held-out split.

### RQ4: Single-node systems efficiency

Does AReno improve a selected efficiency outcome over a faithful public
baseline at matched correctness and task quality?

- **H4:** AReno improves the preregistered primary systems metric while passing
  all correctness gates and meeting the matched quality floor.
- **Falsifier:** The efficiency threshold is not met, the quality floor is
  violated, or the comparison cannot be made under equal information and
  resource budgets.

RQ3 and RQ4 remain unopened until RQ1 and RQ2 pass.

## 3. Claim ladder

| Level | Evidence required | Maximum admissible claim |
| --- | --- | --- |
| L0 | Documentation and code inspection | The repository exposes the described interface or workflow. |
| L1 | Deterministic unit/reference comparisons | The tested component matches the frozen reference on the tested cases. |
| L2 | Bounded end-to-end smoke and one real step | The selected path executes successfully in the recorded environment. |
| L3 | Frozen multi-seed held-out evaluation | The selected training setup improves the specified outcome on the frozen task distribution. |
| L4 | Matched public-baseline comparison | AReno improves the specified metric under the exact matched conditions. |

No result may be described above its achieved level.

## 4. Stage gates

### P0: Novelty, substitute, and protocol audit

**Purpose:** Establish that the proposed scientific question is not already
answered by a direct public substitute and freeze the experiment before
expensive execution.

**Allowed:**

- literature and public-code search;
- read-only source inspection;
- preparation of manifests, reference functions, and protocol tests;
- CPU-only deterministic checks that do not consume a formal outcome.

**Required outputs:**

- `research/p0/search_log.md`;
- `research/p0/candidate_matrix.csv`;
- `research/p0/substitute_audit.md`;
- `research/p0/protocol_freeze.md`;
- `research/p0/source_manifest.json`;
- `research/p0/gate_decision.md`;
- for `PASS_NOVELTY_CONDITIONAL`, one selected primary track and task plus
  explicit model, dataset, split, metric, baseline, seed, tolerance, and sample
  contracts with no unresolved `TBD` fields.

Execution contracts are not required after a terminal kill because no
experiment is admitted. They must be marked `NOT_APPLICABLE_AFTER_TERMINAL_P0`
rather than left as unresolved decisions.

**Allowed terminal outcomes:**

- `PASS_NOVELTY_CONDITIONAL`;
- `KILL_DIRECT_SUBSTITUTE`;
- `BLOCKED_FULL_TEXT_OR_ARTIFACT`;
- `BLOCKED_PROTOCOL_INCOMPLETE`.

P1 cannot start unless P0 returns `PASS_NOVELTY_CONDITIONAL`.

### P1: Deterministic correctness gate

**Purpose:** Verify task, reward, data, algorithm, and checkpoint contracts
before model training.

**Required checks:**

- raw and normalized dataset schemas;
- deterministic task transitions and legal-action checks;
- reward oracle tests for invalid, partial, successful, and optimal outcomes;
- no answer, hidden-state, test, or reward leakage into model-visible inputs;
- masks and message/tool-call ordering;
- analytical or independent-reference loss and gradient comparisons;
- save/load metadata and checkpoint-key coverage where applicable;
- two independent reruns of all deterministic artifacts.

**Required outputs:**

- `research/p1/test_report.txt`;
- `research/p1/artifact_manifest.json`;
- `research/p1/reference_comparison.json`;
- `research/p1/leakage_audit.md`;
- `research/p1/gate_decision.md`.

**GO criteria:**

- all required deterministic tests pass;
- all exact artifacts match where exact equality is required;
- all numerical comparisons pass the frozen tolerance;
- no critical leakage path remains;
- reruns yield the same gate decision.

Otherwise return `KILL_CORRECTNESS` or `BLOCKED_IMPLEMENTATION`.

### P2: Bounded integration pilot

**Purpose:** Determine whether a real model path is executable and whether the
reward signal is non-degenerate. This stage is diagnostic, not confirmatory.

**Prerequisites:**

- explicit owner authorization for model/dataset access and GPU use;
- P1 pass;
- recorded model and dataset licenses;
- exact command and resource ceiling;
- pilot data disjoint from the frozen formal test.

**Execution order:**

1. environment and capacity checks;
2. bounded dataset inspection;
3. smoke inference or smoke training only if justified;
4. one bounded real rollout;
5. inspect one complete trajectory and reward explanation;
6. one real optimizer step;
7. bounded pilot of the preregistered maximum size.

**Required outputs:**

- command and environment manifest;
- raw metrics event files;
- bounded trajectory samples;
- failure and retry log;
- pilot-only analysis;
- `research/p2/gate_decision.md`.

**GO criteria to P3:**

- the complete path executes without protocol repair;
- reward is not constant and is not dominated by parser failure;
- invalid-trajectory, filtering, and timeout rates remain below the thresholds
  frozen in P0;
- the primary metric has enough headroom to distinguish failure and success;
- no pilot observation requires changing the frozen formal outcome.

Any protocol change returns the project to P0/P1 and creates a new version. A
consumed pilot must not be relabeled as formal evidence.

### P3: Frozen formal evaluation

**Purpose:** Test the frozen hypothesis without tuning on the formal outcome.

**Rules:**

- freeze code commit, model checksum, dataset manifest, prompt template,
  tokenizer, decoding settings, topology, seeds, metrics, and analysis script;
- evaluate base policy and no-update control before interpreting trained-policy
  results;
- use the same evaluation budget for every model condition;
- keep training, development, and formal-test artifacts physically separated;
- record every exclusion and failure;
- never rerun selectively to replace an unfavorable result.

**Minimum comparison set:**

1. base policy with frozen decoding;
2. no-update control through the same rollout/evaluation path;
3. trained policy;
4. deterministic or oracle policy when one exists, reported as a diagnostic
   ceiling rather than a deployable baseline;
5. the strongest equal-information baseline selected in P0.

**Primary learning outcome:**

`heldout_task_success_rate`

P0 must define task success, evaluation episode count, training-seed count,
evaluation seeds, and the minimum practically important improvement.

**Secondary outcomes:**

- environment reward;
- legal-action or valid-tool-call rate;
- complete-process compliance;
- timeout, filtering, and invalid-trajectory rates;
- turns, generated tokens, and tool calls per episode;
- latency and peak memory as descriptive metrics only.

**Analysis:**

- report per-seed values and aggregate estimates;
- report effect size and uncertainty interval;
- keep exploratory subgroup analyses separate;
- correct or clearly label multiple comparisons;
- include all preregistered outcomes, including negative results.

**Terminal outcomes:**

- `PASS_FORMAL_PRIMARY`;
- `FAIL_FORMAL_PRIMARY`;
- `E0_INVALID`;
- `BLOCKED_FORMAL_EXECUTION`.

An `E0_INVALID` run is archived and not repaired, spliced, or interpreted as
scientific evidence.

### P4: Ablation and systems comparison

P4 opens only after a valid P3 result.

**Required learning ablations:**

- reward outcome component removed;
- reward process component removed when present;
- no group-relative update or matched alternative loss;
- matched-token or matched-step control;
- context/tool-history ablation for agentic tasks;
- decoding sensitivity analysis.

**Required systems controls:**

- identical model and checkpoint;
- identical dataset records and order;
- identical algorithm semantics and optimizer;
- identical precision, topology, sequence limits, and batch/sample demand;
- identical task-quality floor;
- steady-state measurement after warmup;
- throughput, end-to-end step time, and peak allocated/reserved memory reported
  separately.

Correctness must be revalidated after any implementation or kernel change.
Performance is never accepted as correctness evidence.

### P5: Claim and release gate

Before a scientific report:

- rebuild every table and figure from immutable artifacts;
- verify that reported commands match manifests;
- trace every claim to a passed gate;
- separate diagnostic, controlled, and deployable evidence;
- disclose failed, invalid, blocked, and unopened work;
- conduct an independent rerun or reviewer audit of the primary result.

## 5. Experimental design contract

### 5.1 Data splitting

The selected task must provide:

- a training split;
- a development split used for implementation and selection;
- a frozen formal-test split created before training;
- split-level hashes and generation parameters;
- duplicate and near-duplicate checks;
- hidden-state and solution leakage checks.

Generated environments must use disjoint seed ranges and, where seed identity
alone is insufficient, disjoint structural templates. A test record becomes
exposed when its prompt, hidden state, solution, reward explanation, or model
output is inspected. Exposed records cannot return to the formal test.

### 5.2 Randomness and repeatability

P0 must freeze:

- dataset-generation seeds;
- model initialization or training seeds;
- rollout sampling seeds where controllable;
- evaluation seeds;
- deterministic backend settings;
- all known nondeterministic operations and their mitigation.

The formal analysis must include every frozen seed. Optional stopping based on
intermediate results is prohibited.

### 5.3 Baseline fairness

Every baseline must receive:

- the same model-visible task information;
- the same environment and legal-action interface;
- the same decoding and context budget unless the difference is the
  preregistered intervention;
- the same number of evaluation episodes;
- a documented public implementation commit or a small independent reference
  implementation.

Privileged or oracle baselines must be labeled as diagnostic ceilings.

### 5.4 Metric integrity

The task's primary metric must:

- measure task outcome rather than training loss;
- be computable without model-dependent judgment;
- have one frozen implementation;
- fail closed on malformed trajectories;
- produce an episode-level audit record.

Training reward is not automatically the primary evaluation metric. If reward
is used as an outcome, P0 must demonstrate that it cannot be increased through
formatting, truncation, illegal actions, or reward-component exploitation.

## 6. Artifact and provenance contract

Recommended research layout:

```text
research/
  p0/
    search_log.md
    candidate_matrix.csv
    substitute_audit.md
    protocol_freeze.md
    source_manifest.json
  p1/
    test_report.txt
    artifact_manifest.json
    reference_comparison.json
    leakage_audit.md
    gate_decision.md
  p2/
    run_manifest.json
    commands.txt
    metrics/
    trajectories/
    gate_decision.md
  p3/
    preregistration.md
    frozen_manifest.json
    raw/
    derived/
    analysis/
    gate_decision.md
  p4/
    ablations/
    systems/
  reports/
```

Every run manifest must record:

- protocol version and stage;
- git commit and dirty status;
- command and environment variables excluding secrets;
- model, tokenizer, dataset, and implementation sources with checksums;
- Python, PyTorch, CUDA, driver, GPU, and dependency versions;
- algorithm, topology, precision, optimizer, sequence, batch, and sampling
  settings;
- random seeds;
- input and output artifact hashes;
- start/end time and exit status;
- deviations, retries, exclusions, and gate decision.

Secrets belong in environment configuration and must never appear in this
document, a manifest, a command log, or chat.

## 7. Existing project evidence and boundaries

Use these files as engineering evidence:

- `README.md`: product scope and smoke examples;
- `CODEMAP.md`: code ownership and verification entry points;
- `docs/cli/training.rst`: current CLI behavior;
- `docs/cli/observability.rst`: available runtime metrics;
- `.agents/skills/areno-run-training/SKILL.md`: safe training workflow;
- `.agents/skills/areno-validate-correctness/SKILL.md`: baseline comparison
  workflow;
- `.agents/skills/areno-build-agentic-workflow/SKILL.md`: agentic trajectory
  contract;
- `tests/`: deterministic CPU regression evidence;
- `examples/agentic/`: candidate tasks, not formal experimental protocols.

Known documentation issues that must be closed before P1:

1. The short agentic quickstart omits CLI-required checkpoint and dataset
   arguments.
2. Reward-function concept/reference pages describe a batch-style signature,
   while current examples and trainer code use one `RewardRecord` per call.
3. DuelGrid's illustrative before/after claim lacks a frozen command,
   checkpoint, split, seed, evaluation script, and raw metric manifest.

These are protocol risks because a formal experiment must use one unambiguous
interface and one reproducible execution path.

## 8. Current gate and next action

### Track 1: unified training/rollout systems claim

P0 completed on 2026-07-28 with `KILL_DIRECT_SUBSTITUTE`. This branch remains
terminal. Its authoritative record is:

- `research/p0/search_log.md`;
- `research/p0/candidate_matrix.csv`;
- `research/p0/substitute_audit.md`;
- `research/p0/protocol_freeze.md`;
- `research/p0/source_manifest.json`;
- `research/p0/gate_decision.md`.

VeXact and other close public systems make the broad self-contained
training/rollout claim insufficiently differentiated. Changing only wording,
model, task, or scale does not reopen Track 1.

### Track 2: post-selection group-relative advantage distortion

A materially different question began with a new P0 under
`research/p0_trajectory_censoring/` and returned
`PASS_NOVELTY_CONDITIONAL`. The frozen CPU-only G0 under
`research/p1_trajectory_censoring/` then returned `PASS_G0_MECHANISM`.

The admissible claim is narrow:

> Under the frozen source-equivalent estimator and synthetic prompt groups,
> deleting over-context trajectories before group normalization can change the
> magnitude or sign of survivor advantages and can collapse small survivor
> groups.

Primary evidence:

- `research/p0_trajectory_censoring/protocol_freeze.md`;
- `research/p0_trajectory_censoring/gate_decision.md`;
- `research/p1_trajectory_censoring/g0_results.json`;
- `research/p1_trajectory_censoring/test_report.txt`;
- `research/p1_trajectory_censoring/reference_comparison.json`;
- `research/p1_trajectory_censoring/gate_decision.md`;
- `research/p1_trajectory_censoring/hypothesis_report.tex`.

Two independent process runs produced byte-identical G0 JSON with SHA-256
`ee4aeba6a362107bbc71a85a69dc9850468a8e2d9620568acca2e39f08631146`.

The user opened `P2a_CPU_RUNTIME_PARITY` on 2026-07-28. In a minimal isolated
CPU environment, the actual pinned NumPy advantage function, context filter,
and policy-only materialization method passed all frozen checks. The actual
function/reference comparison covered 40,000 values with zero failures and
maximum absolute error `3.3885e-7`. Two independent P2a processes produced
byte-identical JSON with SHA-256
`443ee345ba83bf6f9ada038f752cef216276f46c4c451b427d12efcb3f0acfbe`.

P2a evidence:

- `research/p2a_trajectory_censoring/protocol_freeze.md`;
- `research/p2a_trajectory_censoring/run_p2a.py`;
- `research/p2a_trajectory_censoring/p2a_results.json`;
- `research/p2a_trajectory_censoring/test_report.txt`;
- `research/p2a_trajectory_censoring/gate_decision.md`;
- `research/p2a_trajectory_censoring/artifact_manifest.json`.

The user opened `P2b_EMPIRICAL_PREVALENCE` on 2026-07-28. TC-P2b-v1 used the
official ModelScope `Qwen/Qwen3-0.6B` snapshot, 16 frozen Codebreaker
trajectories, direct bounded CUDA inference, and the actual AReno tokenizer,
tool parser, trajectory assembly, context filter, group normalization, and
materialization functions. It returned
`FAIL_P2B_NO_PREVALENCE_IN_BOUNDED_PILOT`.

All 16 trajectories were 2095–2112 tokens. Each prospectively frozen
operational cap (512, 768, 1024, and 1536) therefore filtered 16/16 rows. There
were no survivors and no partially censored prompt group, so the frozen
advantage-distortion prevalence gate could not pass. The model outputs were
otherwise structurally usable: parser failures were zero and rewards were
nonconstant. This is a valid negative bounded pilot, not an invalid run and not
evidence that the deterministic mechanism is absent.

P2b-v1 evidence:

- `research/p2b_trajectory_censoring/protocol_freeze.md`;
- `research/p2b_trajectory_censoring/run_p2b.py`;
- `research/p2b_trajectory_censoring/p2b_results.json`;
- `research/p2b_trajectory_censoring/trajectories.jsonl`;
- `research/p2b_trajectory_censoring/model_asset_manifest.json`;
- `research/p2b_trajectory_censoring/test_report.txt`;
- `research/p2b_trajectory_censoring/execution_log.md`;
- `research/p2b_trajectory_censoring/gate_decision.md`.

The canonical P2b result SHA-256 is
`3f021cef36e03d7c986327ea02eb0fbee487118296b211237b25fa85adf7988d`.
This run used real-model trajectories plus an AReno function chain, not the
full AReno CUDA engine; no optimizer step or weight update occurred.

No mitigation experiment opens until P2 shows that the mechanism occurs in an
empirical workload. No real-model learning-harm claim is admitted.

```text
TRACK1_P0_SYSTEMS_DIRECTION: KILL_DIRECT_SUBSTITUTE
TRACK2_P0_NOVELTY: PASS_NOVELTY_CONDITIONAL
TRACK2_G0_ESTIMATOR_MECHANISM: PASS
TRACK2_G0_REPRODUCIBILITY: PASS_BYTE_IDENTICAL
TRACK2_P2A_RUNTIME_PARITY: PASS_ISOLATED_ACTUAL_FUNCTIONS
TRACK2_P2A_REPRODUCIBILITY: PASS_BYTE_IDENTICAL
TRACK2_P2B_V1_EMPIRICAL_PREVALENCE: FAIL_VALID_NEGATIVE_ALL_ROWS_FILTERED
TRACK2_MITIGATION: NOT_ADMITTED
TRACK2_REAL_MODEL_LEARNING_CLAIM: NOT_ADMITTED
```

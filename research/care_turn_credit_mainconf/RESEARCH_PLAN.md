# CARe: Complete Main-Conference Research Plan

Status: `P0_PASS_P1_PASS_P2_PASS_P3_CPU_FREEZE_GPU_BLOCKED`
Working title: **CARe: Risk-Controlled Abstention for Turn-Level Credit in
Agentic Reinforcement Learning**

## 1. Answer first

The issue #199 ablation has research value as an instrument, but not as a
main-conference method. A viable method paper must move from static
`all_assistant / last_assistant / final_answer` masks to a signed per-turn
credit method with a falsifiable reliability claim.

The proposed route is CARe:

- obtain a cheap signed turn-credit estimate from an existing scorer;
- audit a small, prespecified subset with counterfactual environment branches;
- calibrate a trajectory-block risk threshold;
- abstain from policy updates on uncertain turns;
- route the remaining signed credit under a fixed supervision-token budget.

The main contribution is not a new semantic judge. It is risk control over
wrong-sign credit under on-policy drift, plus a cost-aware evaluation that
counts the counterfactual audits.

## 2. Intended paper contributions

Subject to P0 and experiments, the paper would aim to contribute:

1. A formulation of **selective turn credit with abstention**, distinguishing
   missing credit from confidently positive or negative credit.
2. A trajectory-block calibration procedure for bounding wrong-sign credit
   risk within each frozen behavior-policy iteration.
3. A budgeted routing rule that separates attribution quality from supervision
   density.
4. A controlled bifurcation benchmark with exact causal turn labels.
5. Multi-task evidence and complete accounting of environment calls,
   trainable tokens, masked tokens, GPU-hours, and wall clock.

Claims about lower compute are forbidden unless a later implementation actually
skips dense forward/backward work. Loss masking alone changes gradients, not
necessarily FLOPs.

## 3. Method specification

### 3.1 Target credit

At pre-turn state `s_t`, the policy chooses action `a_t`. Define the target
credit as the advantage of that action relative to a frozen counterfactual
action distribution under the same behavior policy:

```text
c_t = E[R | s_t, a_t] - E_{a' ~ pi_old(.|s_t)}[R | s_t, a']
```

The environment state must be checkpointable or exactly replayable. The
counterfactual target is estimated only for a calibration subset, with a
frozen branching count and randomization scheme.

### 3.2 Cheap scorer

CARe is evaluated with at least two scorer families:

- a verifier/potential scorer where gold or intermediate checks exist,
  comparable to TRACE/VPR;
- a structured semantic scorer comparable to TRIAGE on less-verifiable turns.

The scorer outputs magnitude, sign, and uncertainty features. CARe does not
claim the scorer itself as novel.

### 3.3 Block risk calibration

Turns within one trajectory are dependent. Calibration therefore treats the
trajectory, not the turn, as the exchangeable unit. Before thresholding,
confidence-ranked turns enter a fixed, deterministic prefix whose declared
gradient-token mass is at most budget `B`; the prefix is not backfilled. For a
strictness threshold `lambda`, the bounded trajectory loss is the wrong-sign
gradient mass among the nested selected turns divided by the fixed budget
`B`.

This corrects the earlier proposal to guarantee the wrong-sign fraction among
selected turns. That conditional fraction is generally non-monotone in the
threshold, so standard conformal risk control does not apply to it. It remains
a diagnostic without the primary theorem.

At each calibration refresh:

1. freeze `pi_old`;
2. sample disjoint calibration and update trajectories from the same task
   mixture;
3. compute sparse counterfactual labels only on calibration trajectories;
4. choose the most permissive threshold satisfying the frozen conformal-risk
   rule;
5. apply it to update trajectories from that same behavior-policy batch;
6. update the policy only after selection is frozen.

No guarantee is claimed across arbitrary policy or task shift. Refresh
frequency and policy KL are explicit ablations.

### 3.4 Budgeted routing

After risk filtering, admit turns by lower-confidence-bound credit per
trainable token until budget `B` is reached. The primary budget is expressed as
assistant trainable tokens per policy update.

Required controls use the exact same `B`:

- random turns;
- last turn;
- top absolute-credit turns without calibration;
- all turns with loss rescaled to match total weight;
- no-update control.

### 3.5 Policy objective

The base optimizer remains GSPO/GRPO-like. Each assistant token inherits the
signed weight of its parent turn. Sequence normalization must be specified
independently of the loss mask so that selecting fewer tokens does not silently
change objective scale.

Implementing this requires per-turn offsets and per-turn advantages. The
current static `LossMaskPolicy` is insufficient. Any public config or CLI
change requires a separate user decision under `AGENTS.md`.

## 4. Experimental program

### 4.1 Task families

| Role | Candidate | Purpose | Admission condition |
|---|---|---|---|
| Mechanism | Controlled bifurcation suite | Exact causal turn labels, calibration stress tests | CPU oracle and replay exactness |
| Short/medium tool use | WebShop | Comparable to prior agentic CA work | ModelScope asset or user-provided local path |
| Partially observable planning | ALFWorld | Different action/observation structure | ModelScope asset or user-provided local path |
| Long-horizon search | SearchGym preferred; BrowseComp-Plus fallback | Contemporary transfer and horizon stress | Reproducible offline corpus and resource gate |
| Engineering only | AReno Shopping and Tic-Tac-Toe | Instrument/runtime checks | Never counted as primary paper evidence |

At least three scientific task families must survive asset and executability
gates. Do not silently fall back to Hugging Face when ModelScope resolution
fails.

### 4.2 Models

- Qualification: `Qwen/Qwen3-1.7B` through ModelScope.
- Primary: `Qwen/Qwen3-4B` through ModelScope.
- Scale robustness: a larger supported checkpoint only if P5 passes and the
  approved budget permits.

The exact snapshot identifiers and content hashes are frozen before any GPU
run. A 0.6B checkpoint is acceptable for runtime smoke tests only.

### 4.3 Algorithms and baselines

Minimum baseline set:

1. base/no-update evaluation;
2. outcome-only GSPO/GRPO over all assistant turns;
3. static final/last-turn selection;
4. random matched-token routing;
5. magnitude-only top-k routing;
6. one strong turn-level baseline such as TRACE, TRIAGE, or MT-GRPO;
7. CARe applied to the identical base scorer.

If reproducible public code for the strongest direct baseline cannot be adapted
faithfully, the task is `BLOCKED_BASELINE`, not a baseline failure.

### 4.4 Ablations

- target risk `alpha`: `0.05, 0.10, 0.20`;
- token budget: `25%, 50%, 100%` of all-assistant trainable tokens;
- calibration refresh: every `1, 2, 4` policy iterations;
- audit fraction and counterfactual branch count;
- block versus naive turn-i.i.d. calibration;
- positive-only versus signed routing;
- scorer family;
- horizon and injected scorer-noise level;
- fixed normalization versus current mask-length normalization.

Only a small qualification grid may tune these. The confirmatory setting is
frozen before held-out outcomes.

## 5. Outcomes and estimands

### Primary outcomes

- strict held-out task success or frozen normalized return;
- trajectory-block wrong-sign gradient-mass risk;
- area under held-out success versus **total environment transitions**,
  including audit branches.

### Mandatory secondary outcomes

- per-step reward mean and dispersion;
- `trainable_tokens` and `masked_response_tokens`;
- selected-turn coverage and abstention rate;
- positive/negative sign precision and recall;
- valid/executable call rate by turn;
- policy KL, clipping fraction, gradient norm, and non-finite events;
- environment calls, generated tokens, GPU-hours, wall clock, and peak memory.

### Primary comparison

`CARe(base scorer)` versus `uncalibrated base scorer`, under:

1. matched total environment interactions; and
2. matched trainable-token budget.

Step-matched curves are descriptive only.

## 6. Statistical plan

- Training seed is the independent replication unit.
- Use three seeds only for qualification; use at least five independent seeds
  for each primary confirmatory arm unless pilot variance requires more.
- Use paired evaluation tasks within seeds.
- Estimate task effects with a hierarchical bootstrap that resamples seeds,
  then tasks within seed.
- Report all seed trajectories; never select the best checkpoint by held-out
  test performance.
- Use validation-only checkpoint selection with a frozen rule.
- Correct the three task-family primary contrasts with Holm's procedure.
- Report absolute effects, 95% intervals, and the preregistered negligible
  region, not only p-values.
- Invalid trajectories remain failures in strict-success denominators.
- Missing/failed runs follow a frozen intention-to-train policy; hardware
  failures may be rerun only from the same predeclared seed and checkpoint with
  the failure artifact retained.

The initial practical threshold is 5 absolute success points. P3 may change it
using qualification variance and cost, but the change must occur before P5 and
must not use held-out outcomes.

## 7. Stage plan and deliverables

### P0 — Full-text novelty audit, 1–2 weeks, CPU

Deliver:

- direct-substitute matrix and source manifest;
- method residual-gap statement;
- baseline code availability audit;
- terminal `PASS`, `KILL`, or `BLOCKED` decision.

No training, model download, or paid API.

### P1 — Formalization and exact oracle, 2–3 weeks, CPU

Deliver:

- formal block-risk theorem and assumptions;
- controlled bifurcation environment;
- exact leave-one-turn-out/counterfactual labels;
- simulations showing nominal versus empirical risk under independence,
  within-trajectory dependence, and policy shift;
- tests that fail for naive turn-level calibration.

Kill if the guarantee requires unrealistic exchangeability or cannot be
distinguished from ordinary threshold tuning.

### P2 — AReno instrument qualification, 2–4 weeks, CPU

Deliver:

- preserved per-turn token offsets;
- signed per-turn advantage/weight path;
- explicit normalization independent of selected length;
- artifact schema and deterministic exporter;
- seed provenance from initialization through environment;
- invalid-call rejection with no synthesized tool actions;
- CPU tests for zero-update, sign routing, budget, and schema.

Ask before changing public configs or CLI.

### P3 — Bounded GPU executability pilot, about 1 week

Requires explicit user authorization after presenting:

- exact commit and clean diff;
- exact remote commands;
- ModelScope asset manifests;
- one checkpoint, one task, two arms, three qualification seeds;
- hard step/time/GPU-hour ceiling.

This pilot measures executability, variance, throughput, memory, and artifact
completeness. It is not efficacy evidence.

The frozen provider route is one AutoDL A800 80GB. Remote work is split into
two authorization boundaries:

1. read-only inventory, environment installation, checkout, CPU tests, model
   download/hash verification, dataset preparation, and dry preflight;
2. the six GPU training commands, authorized only after the first boundary
   reports a clean commit, accepted asset hashes, live price, and resource
   inventory.

The exact metadata handoff and stop conditions are in
`p3/AUTODL_HANDOFF.md`.

### P4 — Power and resource freeze, about 1 week, CPU

Use only P3 qualification data to freeze:

- seed count;
- steps/interactions;
- audit rate and branch count;
- arm set;
- task split;
- exclusion and rerun rules;
- total GPU-hour ceiling.

Return `KILL_UNDERPOWERED_OR_UNAFFORDABLE` if the confirmatory design does not
fit the approved ceiling.

### P5 — Confirmatory study, 4–7 weeks

Execute the immutable manifest without selective reruns. The minimum paper
matrix is:

- two model scales;
- three scientific task families;
- CARe, its uncalibrated scorer, outcome-only, random matched-budget, and
  magnitude-only matched-budget;
- at least five seeds for the primary arms.

Lower-priority baselines may use fewer seeds only if declared exploratory.

### P6 — Robustness and transfer, 2–3 weeks

Open only after P5:

- unseen horizon/task strata;
- scorer shift;
- calibration refresh shift;
- one contemporary long-horizon environment;
- cost-quality frontier.

No task-specific retuning on test data.

### P7 — Paper and public artifact, 3–4 weeks

Deliver:

- paper, appendix, code, frozen configs, manifests, raw structured metrics;
- environment and asset provenance;
- exact reproduction commands;
- negative/null results and failed seeds;
- reviewer-style claim audit and reproducibility checklist.

## 8. Resource plan

Use GPU-hours, not currency, until actual provider rates are selected.

| Tier | Purpose | Provisional envelope | Decision |
|---|---|---:|---|
| CPU | P0–P2 | local | Authorized |
| GPU-Q | P3 executability | 1× AutoDL A800 80GB; ≤6 training GPU-hours, ≤8 billed hours, ≤CNY 60 | Ask first |
| GPU-P | limited variance pilot | 40–120 GPU-hours | Ask first after GPU-Q |
| GPU-C | confirmatory 1.7B/4B matrix | provisional 400–1,000 GPU-hours | Freeze from measured throughput |
| GPU-X | larger-scale/long-horizon extension | not budgeted | Open only after P5 pass |

These are planning envelopes, not quotes. The confirmatory envelope can change
substantially with horizon, group size, counterfactual branch count, and AReno
throughput. Rent nothing until GPU-Q benchmarks the real path.

## 9. Timeline and venue strategy

From `2026-07-29`, a credible route needs roughly 20–24 weeks:

- Weeks 1–2: P0;
- Weeks 3–6: P1;
- Weeks 7–10: P2;
- Weeks 11–12: P3/P4;
- Weeks 13–19: P5;
- Weeks 20–21: P6;
- Weeks 22–24: paper and artifact.

ICLR 2027 has an official conference page but no verified submission dates in
this audit; using the 2026 September schedule as if it were confirmed would be
unsafe. The realistic primary planning target is **ICML 2027** if its future
official deadline leaves enough time, with **NeurIPS 2027** as the fallback.
For an NLP-centered framing, ACL 2027 is also plausible, but its official site
currently lists submission dates as TBA:
[ACL 2027](https://2027.aclweb.org/).

Venue choice is frozen only after P3 shows the method is executable and the
official calls are available.

## 10. Remote GPU operating contract

All source changes happen locally on a dedicated branch, are tested, committed,
and pushed. The remote host only fetches and checks out that exact commit.
Never copy uncommitted source to the remote host.

Current exact CPU preparation and issue #199 smoke commands already exist in:

`examples/agentic/trainable_turns_ablation/README.md`

The safe current remote preparation command is:

```bash
python examples/agentic/trainable_turns_ablation/run_ablation.py \
  --run-root artifacts/issue-199-ablation \
  --count 32 \
  --dataset-seed 2026 \
  --max-steps 10 \
  --prepare
```

It does not train. The existing explicit GPU command must still be presented to
the user for approval before use.

CARe training commands do not yet exist and must not be fabricated in this
plan. P2 will create and CPU-test the runner; P3 then freezes the exact,
placeholder-free remote commands, commit, asset hashes, and ceiling for user
approval.

Remote prerequisites remain:

```bash
python -m pip install psutil flash-linear-attention
python -m pip install -e . --no-build-isolation
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
areno env --json
areno check
```

Use `--model-hub modelscope` for repository model references. If
`areno/accel` has not changed, do not reinstall AReno merely to update Python
research files.

## 11. Reviewer pre-mortem

Likely rejection reasons and required defenses:

| Reviewer objection | Required evidence |
|---|---|
| “This is just thresholded TRACE/TRIAGE.” | Formal risk target, naive-threshold baseline, and direct-substitute audit |
| “Conformal assumptions fail on-policy.” | Trajectory blocks, same-policy calibration/update split, refresh/KL ablation |
| “Gains come from fewer tokens.” | Random and magnitude-only matched-token controls |
| “You hide counterfactual cost.” | Total environment-call and GPU-hour accounting |
| “Static masks already answer this.” | Show that static arms cannot express signed credit or abstention |
| “Only toy environments.” | Three task families including contemporary long-horizon transfer |
| “No real compute saving.” | Avoid FLOP claims unless sparse execution is implemented and measured |
| “Too many flexible choices.” | Immutable P5 manifest and frozen analysis code |

## 12. Claim ladder

Before P3:

> We have identified a conditional research gap and specified a falsifiable,
> risk-controlled turn-credit method.

After a valid P3:

> The proposed path executes and produces complete artifacts under the recorded
> qualification setting.

Only after P5:

> Under the tested models, tasks, budgets, and assumptions, CARe controls the
> audited wrong-sign risk and improves or fails to improve held-out agent
> learning relative to matched controls.

No stage permits “state of the art,” general agent reliability, or lower GPU
compute without direct evidence.

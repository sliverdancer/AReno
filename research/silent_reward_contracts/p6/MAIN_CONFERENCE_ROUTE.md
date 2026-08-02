# ARCA dynamic publication route after P6

Decision date: `2026-08-02`

## Answer first

The main-conference route still exists, but the current work is not a
main-track method paper. The defensible immediate target is a TMLR audit/tooling
paper. A main-conference route is open through evaluation or systems
contributions, and is closed for method claims until new downstream evidence
exists.

## Current routes

| Route | Current state | Reason |
|---|---|---|
| TMLR audit/tooling | `PRIMARY` | Rolling, correctness-focused, and compatible with bounded empirical audits; P6 still must pass artifact, compile, and reviewer gates. |
| TMLR to J2C | `CONDITIONAL_AFTER_ACCEPTANCE` | Official TMLR review supports selective Journal-to-Conference certification; authors cannot assume certification. |
| NeurIPS Evaluations & Datasets | `OPEN_NEXT_CYCLE_IF_SCOPE_PERSISTS` | The 2026 track explicitly accepted audits, negative results, and executable evaluation tools, but its deadline passed. |
| NeurIPS/ICML/ICLR main method track | `CLOSED_NOW` | ARCA has no new optimizer, no valid P4 consequence study, and no model-quality result. |
| MLSys-style systems route | `CONDITIONAL_EXTENSION` | Requires runtime integration, overhead, prevented-invalid-run evidence, and operational baselines. |
| NLP findings/demo route | `SECONDARY` | Possible if the paper is reframed around LLM-agent training infrastructure and a usable tool, but weaker than TMLR at present. |

## Dynamic gates

Dynamic planning changes the route only when a gate is met. It never pools
natural and synthetic cases or repairs the invalid P4 protocol.

### Gate E: evaluation-track paper

Open a future evaluation-track submission only if all conditions hold:

1. at least eight independently maintained public systems have a completed,
   source-pinned inspection under a preregistered sampling rule;
2. at least four new natural cases occur on core training paths rather than
   evaluation-only replay, with every selected negative inspection retained;
3. at least 20 independent framework/workload clean-control cells are audited,
   and uncertainty is reported rather than thresholded by the point estimate;
4. two external contributors reproduce cases from the anonymous artifact;
5. the benchmark schema, licensing, maintenance policy, and executable artifact
   meet the target track's then-current rules.

Failure of Gate E does not invalidate TMLR; it keeps the main-conference
evaluation route closed.

### Gate S: systems paper

Open a systems route only if a fail-closed preflight is integrated into at least
two real training stacks and compared with native checks. Report detection
latency, CPU and memory overhead, blocked false positives, prevented GPU-hours,
and behavior under concurrency or asynchronous reward workers. A reasonable
engineering target is median preflight overhead below 1% of setup time, but
this is a design target, not a claim or a post-hoc acceptance threshold.

### Gate M: method paper

Keep the method route closed unless a genuinely new intervention is defined
before results and evaluated under a new protocol across at least two model
families and two task families. The study needs a valid contrast between
contract-preserving and contract-breaking conditions, downstream learning
metrics, multiple seeds, and a causal analysis that separates reward-content
quality from transport integrity. Any such experiment requires explicit GPU
authorization and cannot resume P4-v0.2.

## Recommended sequence

1. Finish P6 and submit to TMLR only after `PASS_P6_TMLR_READY`.
2. During review, perform CPU-only external replication intake under Gate E;
   do not make public disclosure or maintainer contact without authorization.
3. If Gate E passes before the next official call, prepare a distinct
   evaluation-track version only if TMLR dual-submission and overlap rules
   permit it. Otherwise pursue TMLR and possible J2C, not parallel archival
   submission.
4. Rent GPU resources only after a new Gate S or Gate M protocol is frozen and
   separately authorized.

## Kill rules

- Discovery of a direct substitute with equivalent executable coverage and
  stronger natural evaluation triggers `KILL_MAIN_CONFERENCE_NOVELTY` and a
  narrow reproduction/reporting reassessment.
- Failure to obtain broader independent evidence triggers
  `STAY_TMLR_AUDIT_SCOPE`, not a negative scientific result.
- Another infrastructure-invalid GPU protocol closes that protocol; it may not
  be selectively repaired into the same consumed result set.

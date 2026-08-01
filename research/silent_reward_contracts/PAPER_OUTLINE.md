# Paper outline and analysis plan

## Abstract contract

State the problem, six natural cases in five pinned systems, the frozen
OpenRLHF mutation-transfer result, and the five-system P5 external-validation
sample with its clean-control denominator. State that P4 was infrastructure
invalid before model loading. Do not advertise model gains.

## Sections

1. **Introduction** — successful execution is weaker than scientific
   learnability; define a silent contract failure.
2. **Contract taxonomy** — F1–F6 and the trace → reward → advantage → mask →
   artifact chain.
3. **ARCA** — executable rules, positive controls, fail-closed artifacts, and
   runtime cost.
4. **Study design** — discovery/development/held-out split, single-fault
   mutations, source pins, frozen hashes, baselines, and gates.
5. **Natural case studies** — AReno, veRL reward flow, veRL timeout provenance,
   post-freeze AReaL replication, slime eval replay, and rLLM evaluator
   coercion.
6. **Frozen transfer evaluation** — OpenRLHF mutation transfer, P5 natural
   external validation, per-family recall, clean false positives, negative
   inspections, and sensitivity analysis.
7. **Invalid dynamic attempt** — report the terminal pre-model import failure
   and exclude it from scientific results.
8. **Limitations and responsible disclosure** — purposive rather than
   prevalence sampling, wide external-control interval, no
   general reward-hacking defense, mutation dependence, and disclosure policy.

## Primary tables and figures

- Figure 1: contract chain and insertion points;
- Table 1: taxonomy, observable symptom, positive control, severity;
- Table 2: natural cases with source commit and conclusion flip;
- Figure 2: held-out macro recall and clean FPR by baseline, with denominators;
- Table 3: per-rule Wilson intervals and framework/workload cells;
- Table 4: all five P5 candidates, audited boundary, result, and source hash.

## Statistical rules

- Framework × workload × family is the inferential cell; fixture seeds are
  repeated measurements.
- Report macro recall, clean false-positive rate, Wilson 95% intervals, and the
  preregistered hierarchical bootstrap.
- Do not plot or analyze P4 arm differences: no scientific P4 trajectory was
  generated.
- Report the P5 clean point estimate and Wilson interval together; do not imply
  that `0/5` proves an ecosystem FPR below 5%.
- Report all natural and synthetic denominators separately. Missing or failed
  runs are terminal outcomes, never silently excluded.

## Appendix

Include the direct-substitute matrix, full mutation registry, source and
artifact hashes, exact CPU/GPU commands, terminal CARe history, Arbor tree,
claim ledger, and reviewer red-team.

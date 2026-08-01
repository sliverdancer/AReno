# SAS-TR-v2.1 B2 GPU Inference Qualification

Decision: `PASS_B2_INTERFACE_N128`

The capacity-only successor completed all 160 prespecified trajectories in
437.90 GPU-wall seconds. Both non-task serving preflights passed, every request
returned a preserved response object, no call was synthesized, validation used
only the mechanically selected interface, and reserve remained unopened.

## Frozen results

| Cell | First-turn executable | Four-turn complete | Positive reward | Role |
| --- | ---: | ---: | ---: | --- |
| D128 | 0.0% | 0.0% | 0.0% | diagnostic |
| D512 | 87.5% | 31.25% | 6.25% | diagnostic |
| N128 | 100.0% | 100.0% | 25.0% | eligible, selected |
| N512 | 100.0% | 100.0% | 25.0% | eligible fallback |

N128 was selected before validation because it was the first passing interface
in the frozen order. On 32 untouched validation trajectories it achieved 100%
first-turn executability, 100% complete four-turn execution, 34.375% positive
strict reward, zero fabricated calls, and complete raw evidence.

## Interpretation

The paired diagnostic pattern supports the registered thinking-budget
mechanism on this checkpoint/runtime/task: D128 emitted no executable first
call, while D512 recovered 87.5% first calls but completed only 31.25% of the
full protocol. Disabling thinking eliminated that interface bottleneck at both
token budgets. Because N128 and N512 tied, the result supports the frozen
minimal N128 interface rather than a larger response budget.

This is instrument qualification, not AF-versus-LF evidence. The study has not
trained either supervision arm, measured sample efficiency or stability across
training seeds, compared direct baselines, or replicated across models,
runtimes, and benchmarks. The main-conference hook must therefore remain
`STAY_DIAGNOSTIC`. B3 is eligible for a new plan but remains unopened until a
fresh training protocol and explicit GPU-training authorization exist.

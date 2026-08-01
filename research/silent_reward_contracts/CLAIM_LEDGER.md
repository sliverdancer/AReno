# Claim ledger

| Claim | Status | Permitted wording |
|---|---|---|
| Silent contract failures occur beyond AReno | supported in pinned sources | Six natural cases across AReno, veRL, AReaL, slime, and rLLM were reproduced under registered inspections. |
| ARCA transfers to an unseen framework | conditionally supported | Frozen rules detected all registered OpenRLHF mutations with no false positives on source-derived clean controls. |
| ARCA transfers to natural cases in unseen systems | supported with boundary limits | Frozen rules detected two P5 natural cases without rule changes; slime is an eval-replay case and rLLM is an evaluator-coercion case. |
| ARCA detects naturally occurring OpenRLHF bugs | unsupported | The held-out inspection found no natural OpenRLHF failure. |
| ARCA establishes field-wide prevalence | unsupported | Five systems with qualifying cases plus bounded negative inspections are case evidence, not a prevalence sample. |
| ARCA clean FPR is below 5% in the ecosystem | unsupported | P5 observed 0/5 false positives, but the Wilson 95% upper bound is 0.4345. |
| ARCA is ready for a TMLR audit/tooling submission | conditionally supported | P5 opens paper drafting, but anonymous artifact and paper-review gates remain. |
| ARCA is a main-conference method contribution | unsupported | No valid downstream consequence study or new learning method has been demonstrated. |
| ARCA improves training or model quality | unsupported | P4-v0.2 was invalid before model loading and produced no scientific result. |
| CARe works | contradicted as a current route | Historical v0.2 was terminal due degenerate reward; ARCA does not revive it. |
| `last_assistant` differs from `final_answer` here | false for registered fixture | Treat as a declared equivalence control. |
| GPU validation has been attempted | supported with strict boundary | The first v0.2 command failed during reward import; model execution and the hypothesis were not evaluated. |

Paper text must distinguish natural production-path cases, source-derived clean
controls, synthetic single-fault mutations, and the post-freeze replication.
The slime eval-replay case must also remain distinct from core training-path
cases. Do not aggregate these categories into one vulnerability rate.

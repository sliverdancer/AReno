# ARCA: auditing silent reward contracts in agentic RL

Status: `P5_PASS_EXTERNAL_NATURAL_P4_INVALID_INFRASTRUCTURE`

ARCA is a falsification-first audit package for failures that allow an agentic
RL job to execute and emit finite metrics while its scientific treatment is
invalid. It checks typed reward boundaries, trace pairing, reward-key flow,
reward informativeness, grouped-advantage identifiability, treatment identity,
and failure provenance before paid training.

## Answer first

The CPU route passed its frozen P0–P3 gates. Four natural contract failures
were reproduced across three pinned systems:

1. AReno stringified tool arguments silently zeroed a dict-only reward;
2. veRL produced `tool_rewards` while the inspected reward manager consumed
   `reward_scores`;
3. the same veRL path mapped a timeout to numeric zero without structured
   failure status;
4. AReaL tensorization mapped a missing reward to numeric zero without a
   structured missing-reward status.

The one-shot OpenRLHF held-out evaluation found no natural failure. ARCA scored
1.00 macro recall on 40 registered single-fault mutations with 0.00 false
positive rate on 40 source-derived clean controls. This is transfer under a
controlled mutation protocol, not field-wide prevalence or proof that ARCA
prevents reward hacking.

P5 then froze five previously unseen systems before inspection. Direct CPU AST
probes reproduced two additional natural cases: slime's eval-replay path
conflated missing reward with numeric zero, and rLLM's documented lightweight
dict evaluator defaulted a missing reward to zero even when
`is_correct=True`. Agent-R1, RAGEN, and Agent Lightning remain bounded negative
inspections. This strengthens the TMLR audit/tooling route but does not establish
prevalence or a main-conference method contribution.

## Reproduce

```bash
python -m pytest tests/test_silent_reward_contracts_cpu.py -q
python -m pytest tests/test_arca_p5_cpu.py -q
python -m research.silent_reward_contracts.summarize_p3 \
  --output-dir research/silent_reward_contracts/p3/artifacts/summary
```

Exact upstream CPU commands are in `p2/REMOTE_CPU_COMMANDS.md` and
`p3/REMOTE_CPU_COMMANDS.md`; P5 commands are in `p5/REMOTE_CPU_COMMANDS.md`.
The terminal GPU handoff is in
`p4/REMOTE_GPU_COMMANDS.md`; it must not be executed without a new explicit
authorization.

## Evidence map

| Stage | Result | Primary artifact |
|---|---|---|
| P0 novelty | pass, no direct substitute in frozen screen | `p0/direct_substitute_matrix.csv` |
| P1 AReno production reproduction | pass | `p1/artifacts/production_contract_cases.json` |
| P2 cross-system prevalence gate | pass | `p2/artifacts/cross_system_cases.json` |
| P3 frozen transfer | pass with stated mutation limit | `p3/artifacts/summary/gate_summary.json` |
| P4 dynamic consequence | invalid before model load; not evaluated | `p4/GPU_GATE_DECISION_20260801.md` |
| P5 external natural validation | pass with two bounded natural cases in two unseen systems | `p5/artifacts/external_evidence.json` |

Read `CLAIM_LEDGER.md` before reusing any result in a paper or abstract.

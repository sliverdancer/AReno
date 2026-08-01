# ARCA: auditing silent reward contracts in agentic RL

Status: `PASS_P3_CPU_AUDITOR_TO_GPU_AUTHORIZATION_REQUEST`

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

## Reproduce

```bash
python -m pytest tests/test_silent_reward_contracts_cpu.py -q
python -m research.silent_reward_contracts.summarize_p3 \
  --output-dir research/silent_reward_contracts/p3/artifacts/summary
```

Exact upstream CPU commands are in `p2/REMOTE_CPU_COMMANDS.md` and
`p3/REMOTE_CPU_COMMANDS.md`. The prepared-only GPU handoff is in
`p4/REMOTE_GPU_COMMANDS.md`; it must not be executed without a new explicit
authorization.

## Evidence map

| Stage | Result | Primary artifact |
|---|---|---|
| P0 novelty | pass, no direct substitute in frozen screen | `p0/direct_substitute_matrix.csv` |
| P1 AReno production reproduction | pass | `p1/artifacts/production_contract_cases.json` |
| P2 cross-system prevalence gate | pass | `p2/artifacts/cross_system_cases.json` |
| P3 frozen transfer | pass with stated mutation limit | `p3/artifacts/summary/gate_summary.json` |
| P4 dynamic consequence | awaiting GPU and authorization | `p4/GPU_PROTOCOL.md` |

Read `CLAIM_LEDGER.md` before reusing any result in a paper or abstract.

# P1 production-contract reproduction

Decision: `PASS_P1_PRODUCTION_REPRODUCTION_TO_P2`

The CPU fixture calls AReno's production
`_chat_response_message_tool_calls` normalizer, constructs the resulting
`RewardEvent` sequence, and passes it to the frozen CARe bifurcation reward.
No simulated replacement parser is used.

## Results

| Case | Ordinary execution | Production value | Canonical oracle | Interpretation |
|---|---:|---:|---:|---|
| stringified tool arguments | pass | 0.0 | 1.0 | high-severity silent reward corruption |
| issue #199 last/final masks | pass | identical | identical | declared equivalence control, not an undisclosed bug |

The first case is natural to the pinned AReno/CARe integration. The second is
an identifiability check that passes because the harness README explicitly
declares the alias. It is not counted toward the P2 natural-failure prevalence
threshold.

## Reproduction

```bash
/mnt/d/Python_Releases/python.exe -m pytest \
  tests/test_silent_reward_contracts_cpu.py -q

/mnt/d/Python_Releases/python.exe \
  -m research.silent_reward_contracts.generate_p1_artifacts \
  --output-dir research/silent_reward_contracts/p1/artifacts
```

The Windows Python path records the available local CPU environment used for
this worktree; any Python 3.10+ AReno development environment with its existing
dependencies can run the same commands.

## Artifact hashes

- CSV: `6718f8a4581b2529f0591b104c38fcee208afc0ec4fddcdea2eb021e56758cb9`
- JSON: `d5d058e55caea52694ba49007cac1b4de2f5e0e71ce7482eaa166593f62a6a73`

No GPU, model weight, serving process, or external API was used.

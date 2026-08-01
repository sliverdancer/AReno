# Exact P3 CPU reproduction

Run from an AReno checkout at this branch. These commands do not download
model weights or use a GPU.

```bash
set -euo pipefail
UPSTREAM_ROOT=/tmp/arca-upstreams
mkdir -p "$UPSTREAM_ROOT"
git clone https://github.com/OpenRLHF/OpenRLHF.git "$UPSTREAM_ROOT/openrlhf"
git -C "$UPSTREAM_ROOT/openrlhf" checkout bc71bb19464aca306b33080b2d2bb45d154e2f49
git clone https://github.com/inclusionAI/AReaL.git "$UPSTREAM_ROOT/areal"
git -C "$UPSTREAM_ROOT/areal" checkout 62f955c5e0388aebc6fd58c5ad3f8fcf9d7384b8

python -m research.silent_reward_contracts.openrlhf_adapter \
  --openrlhf-root "$UPSTREAM_ROOT/openrlhf" \
  --output research/silent_reward_contracts/p3/heldout_cases.json
python -m research.silent_reward_contracts.evaluate_auditor \
  --split heldout \
  --output-dir research/silent_reward_contracts/p3/artifacts/heldout
python -m research.silent_reward_contracts.areal_adapter \
  --areal-root "$UPSTREAM_ROOT/areal" \
  --output research/silent_reward_contracts/p3/artifacts/replication_areal.json
python -m research.silent_reward_contracts.summarize_p3 \
  --output-dir research/silent_reward_contracts/p3/artifacts/summary
python -m pytest tests/test_silent_reward_contracts_cpu.py -q
```

Before accepting the output, compare all upstream commits and source hashes
against `upstream_manifest.json`; the summary generator independently rejects
changes to the three frozen evaluator inputs.

# Remote GPU commands — do not execute before fresh authorization

These are the exact commands to use after the owner supplies a new SSH target,
confirms the clean branch commit, and explicitly authorizes this protocol and
budget. Merely preparing the manifest is CPU-only.

```bash
set -euo pipefail
export REPO_ROOT=/root/AReno
export RUN_ROOT=/root/autodl-tmp/arca-p4-v02
export MODEL_ROOT=/root/autodl-tmp/models/Qwen3-0.6B
cd "$REPO_ROOT"
git fetch origin research/silent-reward-contracts
git checkout --detach origin/research/silent-reward-contracts
test -z "$(git status --porcelain)"
ARENO_BUILD_EXT=0 pip install -e . --no-build-isolation
pip install tensorboard
python -m pytest tests/test_silent_reward_contracts_cpu.py -q

python -m research.silent_reward_contracts.p4.prepare_p4 \
  --run-root "$RUN_ROOT" \
  --ckpt "$MODEL_ROOT"
python -c "import json, os; p=json.load(open(os.environ['RUN_ROOT'] + '/manifest.json')); assert p['protocol']=='ARCA-P4-DYNAMIC-v0.2'; assert p['authorization']=='PREPARE_ONLY_GPU_NOT_AUTHORIZED'; assert p['git_status']==''; assert p['run_order']==['strict-seed-3101','canonical-seed-3101','strict-seed-3102','canonical-seed-3102','strict-seed-3103','canonical-seed-3103']; print(json.dumps(p['ceilings'], indent=2))"
```

Stop here. Model acquisition and every `areno train` command remain blocked
until a new explicit execution authorization. After authorization, execute
only through the fail-closed controller:

```bash
python -m research.silent_reward_contracts.p4.execute_p4 \
  --run-root "$RUN_ROOT" \
  --execute-gpu-training
tar -C "$(dirname "$RUN_ROOT")" -czf "$RUN_ROOT.tar.gz" "$(basename "$RUN_ROOT")"
sha256sum "$RUN_ROOT/manifest.json" "$RUN_ROOT/steps.csv" "$RUN_ROOT/steps.json" "$RUN_ROOT.tar.gz"
```

The environment variables deliberately use explicit data-disk paths. Adjust
them only to match the new instance before preparation; do not change the
frozen arms, seeds, order, steps, or ceilings.

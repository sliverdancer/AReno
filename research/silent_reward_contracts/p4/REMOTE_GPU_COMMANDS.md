# Remote GPU commands — do not execute before fresh authorization

These are the exact commands to use after the owner supplies a new SSH target,
confirms the clean branch commit, and explicitly authorizes this protocol and
budget. Merely preparing the manifest is CPU-only.

```bash
set -euo pipefail
export REPO_ROOT=/root/AReno
export RUN_ROOT=/root/autodl-tmp/arca-p4-v01
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
python -c "import json, os; p=json.load(open(os.environ['RUN_ROOT'] + '/manifest.json')); assert p['protocol']=='ARCA-P4-DYNAMIC-v0.1'; assert p['authorization']=='PREPARE_ONLY_GPU_NOT_AUTHORIZED'; assert p['git_status']==''; assert len(p['commands'])==6; print(json.dumps(p['ceilings'], indent=2))"
```

Stop here. Model acquisition and every `areno train` command remain blocked
until a new explicit execution authorization. After authorization, inspect and
execute the manifest commands in their stored insertion order, one at a time:

```bash
python - <<'PY'
import json, os, shlex, subprocess, time
from pathlib import Path
root = Path(os.environ['RUN_ROOT'])
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['protocol'] == 'ARCA-P4-DYNAMIC-v0.1'
assert manifest['authorization'] == 'PREPARE_ONLY_GPU_NOT_AUTHORIZED'
for run_id, command in manifest['commands'].items():
    started = time.monotonic()
    completed = subprocess.run(shlex.split(command), check=False, timeout=3600)
    row = {'run_id': run_id, 'returncode': completed.returncode, 'wall_seconds': time.monotonic() - started}
    with (root / 'execution.jsonl').open('a') as handle:
        handle.write(json.dumps(row, sort_keys=True) + '\n')
    if completed.returncode != 0:
        raise SystemExit(f'stop after {run_id}: {completed.returncode}')
PY

python - <<'PY'
import json, os
from pathlib import Path
from research.silent_reward_contracts.p4.collect_p4 import load_series, build_rows, write_artifacts
from research.silent_reward_contracts.p4.prepare_p4 import ARMS, SEEDS
root = Path(os.environ['RUN_ROOT'])
manifest = json.loads((root / 'manifest.json').read_text())
series = {(arm, seed): load_series(root / 'runs' / f'{arm}-seed-{seed}' / 'metrics') for arm in ARMS for seed in SEEDS}
rows = build_rows(series)
write_artifacts(rows, root, manifest)
print(json.dumps(rows, indent=2))
PY
tar -C "$(dirname "$RUN_ROOT")" -czf "$RUN_ROOT.tar.gz" "$(basename "$RUN_ROOT")"
sha256sum "$RUN_ROOT/manifest.json" "$RUN_ROOT/steps.csv" "$RUN_ROOT/steps.json" "$RUN_ROOT.tar.gz"
```

The environment variables deliberately use explicit data-disk paths. Adjust
them only to match the new instance before preparation; do not change the
frozen arms, seeds, order, steps, or ceilings.

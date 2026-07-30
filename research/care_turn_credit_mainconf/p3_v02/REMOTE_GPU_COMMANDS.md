# P3-v0.2 exact remote-GPU handoff

Protocol: `CARE-P3-PILOT-v0.2`

Frozen source commit:
`c96bcf2da464dff36593d43c8d29991d4b998059`

Authorization: `GPU_TRAINING_NOT_AUTHORIZED`

Target: one AutoDL A800 80GB. Hard ceilings are 6 training GPU-hours, 8
instance-hours, and CNY 60. A newly rented instance must first pass the live
hardware, price, disk, network, and clean-checkout checks. Renting a host does
not authorize the command in section 5.

## 1. Read-only admission

```bash
python --version
git --version
uname -m
nproc
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi
free -h
df -h
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Stop unless there is exactly one x86-64 A800 with at least 80GB VRAM, at least
8 logical CPUs, 64GB RAM, 60GB free persistent disk, a working CUDA PyTorch
runtime, and a projected bill within both ceilings.

## 2. Exact source checkout

```bash
cd /root/autodl-tmp
git clone https://github.com/sliverdancer/AReno.git care-p3-v02
cd /root/autodl-tmp/care-p3-v02
git fetch origin research/trainable-turns-ablation
git checkout --detach c96bcf2da464dff36593d43c8d29991d4b998059
test "$(git rev-parse HEAD)" = "c96bcf2da464dff36593d43c8d29991d4b998059"
test -z "$(git status --short)"
```

Do not edit source on the GPU host.

## 3. Environment and CPU requalification

```bash
python -m pip install psutil flash-linear-attention
python -m pip install -e . --no-build-isolation

areno env --json
areno check

python -m pytest \
  tests/test_care_turn_credit_cpu.py \
  tests/test_care_p3_design_cpu.py \
  tests/test_trainable_turns_ablation_cpu.py \
  tests/test_metrics_cpu.py \
  tests/test_train_cli_config_cpu.py \
  tests/test_protocol_cpu.py -q

PYTHONPATH=. python -c "from areno.experimental.care.turn_credit import load_turn_credit_fn; fn=load_turn_credit_fn('examples/agentic/care_bifurcation/care_router.py'); assert fn.__name__ == 'route_turn_credit'; print('DYNAMIC_LOADER_PASS', fn.__module__)"
```

Any failure stops the protocol.

## 4. Model, manifest, and no-training preflight

```bash
export CARE_REPO=/root/autodl-tmp/care-p3-v02
export CARE_WORK_ROOT=/root/autodl-tmp/care-p3-v02-work
export CARE_MODEL_DIR=/root/autodl-tmp/care-p3-v02-work/Qwen3-0.6B
export CARE_RUN_ROOT=/root/autodl-tmp/care-p3-v02-work/pilot
mkdir -p "$CARE_WORK_ROOT"

python examples/agentic/care_bifurcation/fetch_modelscope_snapshot.py \
  --local-dir "$CARE_MODEL_DIR" \
  --download

python examples/agentic/care_bifurcation/prepare_p3.py \
  --run-root "$CARE_RUN_ROOT" \
  --ckpt "$CARE_MODEL_DIR" \
  --count 64 \
  --dataset-seed 7301 \
  --max-steps 1 \
  --prepare

python -c "import json, os; p=json.load(open(os.environ['CARE_RUN_ROOT'] + '/manifest.json')); assert p['protocol']=='CARE-P3-PILOT-v0.2'; assert p['git_commit']=='c96bcf2da464dff36593d43c8d29991d4b998059'; assert p['git_status']==''; assert p['dynamic_hook_preflight']['status']=='passed'; assert len(p['commands'])==6; print(json.dumps(p['dynamic_hook_preflight'], indent=2))"

python examples/agentic/care_bifurcation/execute_p3.py \
  --run-root "$CARE_RUN_ROOT"
```

The last command must report a dry run with six validated run IDs. It verifies
the clean commit, complete model hashes, exact protocol, and real production
turn-credit loader, but does not train.

Report the inventory, live price, commit, test result, ModelScope verification,
manifest SHA-256, dataset SHA-256, and dry-run output. Then stop and request a
new explicit GPU authorization.

## 5. GPU execution — blocked

Do not run this command until the user explicitly authorizes P3-v0.2 after
reviewing the section 4 report:

```bash
python examples/agentic/care_bifurcation/execute_p3.py \
  --run-root "$CARE_RUN_ROOT" \
  --execute-gpu-training
```

The runner executes the unchanged order:

1. `care-seed-3101`
2. `uncalibrated-seed-3101`
3. `care-seed-3102`
4. `uncalibrated-seed-3102`
5. `care-seed-3103`
6. `uncalibrated-seed-3103`

It stops on the first failure or one-hour run timeout. Never repair or
selectively rerun a failed seed.

## 6. Collection and shutdown

Only if all six runs complete:

```bash
python examples/agentic/care_bifurcation/collect_p3.py \
  --run-root "$CARE_RUN_ROOT"

sha256sum \
  "$CARE_RUN_ROOT/manifest.json" \
  "$CARE_RUN_ROOT/run_results.json" \
  "$CARE_RUN_ROOT/pilot_steps.csv" \
  "$CARE_RUN_ROOT/pilot_steps.json"

tar -C "$CARE_WORK_ROOT" -czf \
  /root/autodl-tmp/care-p3-v02-artifacts.tar.gz pilot
sha256sum /root/autodl-tmp/care-p3-v02-artifacts.tar.gz
```

Download the archive and stop the instance before the 8-hour billing ceiling.
If any run fails, archive the partial directory instead of collecting or
rerunning. P4 opens only after the mechanical P3 gate passes.

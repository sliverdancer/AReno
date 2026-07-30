# P3 remote-GPU command freeze

Status: `COMMANDS_AND_PROPOSED_QUOTE_FROZEN_AWAITING_REMOTE_INSTANCE`
Authorization: `GPU_TRAINING_NOT_AUTHORIZED`

No command in section 4 may run until the user separately approves the clean
commit, provider/GPU, quote, and resource ceilings.

Proposed quote checked on `2026-07-30`: AutoDL A800 80GB at CNY 4.98/hour on
the [official pricing page](https://www.autodl.com/home). The maximum is six
training hours, eight billed instance-hours including setup, and CNY 60 total.
The live rate, assigned CPU/RAM, and storage charge must be rechecked before
provisioning; stop if any frozen limit would be exceeded.

## 1. Required clean checkout

The remote checkout must be a detached, reviewed commit from
`research/trainable-turns-ablation`. The immutable source identifier is the
commit containing this document. After fetching the branch, detach at the
remote branch tip and record:

```bash
git fetch origin research/trainable-turns-ablation
git checkout --detach origin/research/trainable-turns-ablation
git rev-parse HEAD
git status --short
```

The final command must print nothing. Compare the recorded hash with the
published handoff before continuing. Source files are never edited or copied
piecemeal on the GPU host.

## 2. Environment and ModelScope asset

Run from the repository root:

```bash
python -m pip install psutil flash-linear-attention
python -m pip install -e . --no-build-isolation

python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
areno env --json
areno check

python -m pytest \
  tests/test_care_turn_credit_cpu.py \
  tests/test_care_p3_design_cpu.py \
  tests/test_train_cli_config_cpu.py -q
```

Keep assets and run outputs outside the Git checkout:

```bash
CARE_REPO="$(pwd -P)"
CARE_WORK_ROOT="$(dirname "$CARE_REPO")/care-p3-work"
CARE_MODEL_DIR="$CARE_WORK_ROOT/Qwen3-0.6B"
CARE_RUN_ROOT="$CARE_WORK_ROOT/pilot"
mkdir -p "$CARE_WORK_ROOT"
```

The following is a ModelScope download plus full-file verification, not
training:

```bash
python examples/agentic/care_bifurcation/fetch_modelscope_snapshot.py \
  --local-dir "$CARE_MODEL_DIR" \
  --download
```

The download is accepted only when all 11 frozen files, including the
1,503,300,328-byte `model.safetensors`, match the recorded SHA-256 values.

## 3. CPU preparation and read-only preflight

```bash
python examples/agentic/care_bifurcation/prepare_p3.py \
  --run-root "$CARE_RUN_ROOT" \
  --ckpt "$CARE_MODEL_DIR" \
  --count 64 \
  --dataset-seed 7301 \
  --max-steps 1 \
  --prepare

python examples/agentic/care_bifurcation/execute_p3.py \
  --run-root "$CARE_RUN_ROOT"
```

The second command validates the clean commit, exact six-run set, local model
path, asset hashes, and manifest, then exits without training. To inspect the
exact six `areno train` commands:

```bash
python examples/agentic/care_bifurcation/prepare_p3.py \
  --run-root "$CARE_RUN_ROOT" \
  --ckpt "$CARE_MODEL_DIR" \
  --count 64 \
  --dataset-seed 7301 \
  --max-steps 1
```

The generated `manifest.json` is the immutable command record. Within a seed,
the only treatment differences are the turn-credit config path and metrics
directory.

## 4. GPU command — blocked until explicit approval

After the user has approved this exact execution:

```bash
python examples/agentic/care_bifurcation/execute_p3.py \
  --run-root "$CARE_RUN_ROOT" \
  --execute-gpu-training
```

The runner invokes, sequentially:

1. `care-seed-3101`
2. `uncalibrated-seed-3101`
3. `care-seed-3102`
4. `uncalibrated-seed-3102`
5. `care-seed-3103`
6. `uncalibrated-seed-3103`

Each command is killed after 3,600 seconds. The runner stops on the first
failure or timeout and retains stdout, stderr, exit state, and wall time.
Because there is exactly one GPU and at most six runs, the hard training
ceiling is six single-GPU hours. The instance must also be stopped by eight
billed hours from provisioning, even if setup or download is incomplete.

The core command frozen for every run is:

```text
areno train
  --ckpt <verified absolute CARE_MODEL_DIR>
  --model-hub modelscope
  --dataset-path <CARE_RUN_ROOT>/dataset/bifurcation.jsonl
  --dataset-loader-fn <CARE_REPO>/examples/agentic/care_bifurcation/dataset_loader.py
  --reward-fn-path <CARE_REPO>/examples/agentic/care_bifurcation/reward.py
  --agent-fn <CARE_REPO>/examples/agentic/care_bifurcation/run_agent.py
  --turn-credit-fn-path <CARE_REPO>/examples/agentic/care_bifurcation/care_router.py
  --turn-credit-config-path <CARE_REPO>/examples/agentic/care_bifurcation/{arm}_config.json
  --algo grpo
  --seed {3101|3102|3103}
  --tp-size 1
  --world-size 1
  --batch-size 20
  --n-samples 1
  --mini-bs 5
  --gradient-accumulation-steps 4
  --max-running-prompts 20
  --max-prompt-tokens 512
  --max-new-tokens 64
  --max-context-len 4096
  --agent-timeout-s 900
  --attn-backend native
  --disable-thinking
  --max-steps 1
  --metrics-log-dir <CARE_RUN_ROOT>/runs/{arm}-seed-{seed}/metrics
  --trainable-turns all_assistant
```

Angle-bracket notation above explains the resolved fields; the manifest stores
their absolute, placeholder-free values.

## 5. Collection

Only after all six authorized runs finish:

```bash
python examples/agentic/care_bifurcation/collect_p3.py \
  --run-root "$CARE_RUN_ROOT"

sha256sum \
  "$CARE_RUN_ROOT/manifest.json" \
  "$CARE_RUN_ROOT/run_results.json" \
  "$CARE_RUN_ROOT/pilot_steps.csv" \
  "$CARE_RUN_ROOT/pilot_steps.json"
```

Do not interpret a one-step reward difference as learning. Apply the mechanical
gate in `PROTOCOL.md` and retain every failed or invalid run.

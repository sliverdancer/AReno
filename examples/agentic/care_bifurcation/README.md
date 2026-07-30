# CARe controlled bifurcation pilot

Status: `CPU_QUALIFIED_GPU_NOT_AUTHORIZED`

This example is the bounded P3 executability pilot for CARe. It is not the
issue #199 static-mask ablation and it is not confirmatory efficacy evidence.

## Frozen design

- Task: four model-generated `choose_bit` actions followed by a text summary.
- Outcome: deterministic terminal success from signed weights and a threshold.
- Exact audit: enumerate both current actions and every future binary
  continuation under the frozen uniform continuation policy.
- Split: 10 even-indexed calibration trajectories and 10 odd-indexed update
  trajectories from the same behavior-policy batch.
- Arms: `care` and `uncalibrated`.
- Qualification seeds: `3101`, `3102`, and `3103`.
- Update budget: 256 eligible assistant tokens per update trajectory.
- Calibration target: block wrong-sign gradient mass at `alpha = 0.20`.
- Cost matching: both arms execute the same calibration audits; only CARe uses
  the calibrated abstention threshold.
- Proposed GPU: one AutoDL A800 80GB.
- Runtime ceiling: at most one hour per run, six training GPU-hours, eight
  billed instance-hours, and CNY 60 total.

The built-in cheap scorer has deterministic injected sign errors. It exists to
stress the routing mechanism; it is not a learned scorer and is not a claimed
method contribution.

## CPU-only preparation

The default command prints the six commands but writes nothing:

```bash
python examples/agentic/care_bifurcation/prepare_p3.py
```

To create a deterministic dataset and source/command manifest:

```bash
CARE_WORK_ROOT="$(dirname "$(pwd -P)")/care-p3-work"
CARE_RUN_ROOT="$CARE_WORK_ROOT/pilot"
CARE_MODEL_DIR="$CARE_WORK_ROOT/Qwen3-0.6B"

python examples/agentic/care_bifurcation/prepare_p3.py \
  --run-root "$CARE_RUN_ROOT" \
  --ckpt "$CARE_MODEL_DIR" \
  --count 64 \
  --dataset-seed 7301 \
  --max-steps 1 \
  --prepare
```

Preparation creates `GPU_EXECUTION_NOT_AUTHORIZED`. It does not import PyTorch,
load a model, or start training.

## Frozen ModelScope asset

ModelScope exposed only the mutable `master` branch for
`Qwen/Qwen3-0.6B`. The pilot therefore freezes every expected file size and
SHA-256 in `modelscope_asset.json`. Download and verification are separate from
training:

```bash
python examples/agentic/care_bifurcation/fetch_modelscope_snapshot.py \
  --local-dir "$CARE_MODEL_DIR" \
  --download
```

Running without `--download` performs verification only. Any missing or changed
file is a hard failure.

## GPU execution boundary

`execute_p3.py` refuses dirty checkouts, changed commits, remote checkpoint
references, incomplete manifests, and model files that do not match the frozen
hashes. Without `--execute-gpu-training` it is a read-only preflight:

```bash
python examples/agentic/care_bifurcation/execute_p3.py \
  --run-root "$CARE_RUN_ROOT"
```

Do not add `--execute-gpu-training` until the user has approved the exact clean
commit, provider quote, GPU type, and six commands in the generated manifest.

## Artifact collection

After an authorized, complete six-run pilot:

```bash
python examples/agentic/care_bifurcation/collect_p3.py \
  --run-root "$CARE_RUN_ROOT"
```

The collector writes aligned `pilot_steps.csv` and `pilot_steps.json`. Required
fields include reward mean/dispersion, `trainable_tokens`,
`masked_response_tokens`, selected/masked turn-credit mass, audit calls,
calibration/update block counts, validity, and the calibrated threshold.

## CPU verification

```bash
python -m pytest tests/test_care_p3_design_cpu.py -q
```

The tests use synthetic in-memory trajectories and fake asset files. They do
not download a model or access a GPU.

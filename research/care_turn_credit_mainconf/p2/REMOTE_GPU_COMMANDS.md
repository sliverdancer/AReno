# P2 Remote-GPU Command Freeze

Status: `DOCUMENTED_NOT_EXECUTED`
Authorization: CPU preparation only. Do not run either training command below
without a new explicit user approval after the remote manifest and GPU quote
have been inspected.

## 1. Environment and CPU qualification

Run from a clean checkout containing the reviewed P2 files:

```bash
python -m pip install psutil flash-linear-attention
python -m pip install -e . --no-build-isolation

python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
areno env --json
areno check

python -m pytest \
  tests/test_care_turn_credit_cpu.py \
  tests/test_trainable_turns_ablation_cpu.py \
  tests/test_train_cli_config_cpu.py -q

python examples/agentic/trainable_turns_ablation/run_ablation.py \
  --run-root artifacts/issue-199-ablation \
  --count 32 \
  --dataset-seed 2026 \
  --train-seed 2026 \
  --max-steps 10 \
  --prepare
```

Before any GPU authorization, preserve:

```bash
git rev-parse HEAD
git status --short
sha256sum artifacts/issue-199-ablation/manifest.json
```

The checkout must be clean and its commit must be the reviewed commit recorded
for the run. Source files must not be edited on the remote host.

## 2. Frozen issue #199 three-arm smoke command

This invokes the already documented `all_assistant`, `last_assistant`, and
`final_answer` arms and collects per-step reward and token counts:

```bash
python examples/agentic/trainable_turns_ablation/run_ablation.py \
  --run-root artifacts/issue-199-ablation \
  --count 32 \
  --dataset-seed 2026 \
  --train-seed 2026 \
  --max-steps 10 \
  --execute-gpu-training
```

This is an engineering smoke ablation, not main-conference efficacy evidence.

## 3. Frozen P2 reference-hook qualification command

The following runs the non-method outcome-broadcast control through the real
agentic GRPO path. It exists only to qualify the public hook, signed token
materialization, fixed-budget normalization, and JSONL diagnostics:

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --model-hub modelscope \
  --dataset-path artifacts/issue-199-ablation/dataset/tictactoe.jsonl \
  --dataset-loader-fn examples/agentic/tictactoe/dataset_loader.py \
  --reward-fn-path examples/agentic/tictactoe/reward.py \
  --agent-fn examples/agentic/trainable_turns_ablation/run_agent.py \
  --turn-credit-fn-path examples/agentic/trainable_turns_ablation/outcome_broadcast_turn_credit.py \
  --turn-credit-config-path examples/agentic/trainable_turns_ablation/outcome_broadcast_turn_credit.json \
  --algo grpo \
  --seed 2026 \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --n-samples 4 \
  --mini-bs 4 \
  --max-running-prompts 4 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --max-context-len 2048 \
  --agent-timeout-s 900 \
  --attn-backend native \
  --max-steps 10 \
  --metrics-log-dir artifacts/care-p2-reference/metrics \
  --trainable-turns all_assistant
```

Expected additional artifact:

```text
artifacts/care-p2-reference/metrics/turn_credit_diagnostics.jsonl
```

The outcome-broadcast hook must never be presented as CARe or as a learned,
audited, or novel credit-assignment method.

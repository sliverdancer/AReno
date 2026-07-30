# Issue #199: Three-Arm GPU Ablation

This harness compares `all_assistant`, `last_assistant`, and `final_answer`
with the same checkpoint, deterministic Tic-Tac-Toe dataset, reward, agent,
optimizer settings, and step count. It writes per-step reward mean/std,
`trainable_tokens`, and `masked_response_tokens` to CSV and JSON.

The agent reuses the existing Tic-Tac-Toe dataset loader, game, tool schema,
and reward. It adds a second model call after the real tool result so
`final_answer` always has a trainable text span on well-formed trajectories.
Consequently, `last_assistant` and `final_answer` intentionally select the same
span and form an equivalence-control pair. `all_assistant` additionally trains
the tool-call span.

The default harness invocation is dry-run. GPU work starts only with the
explicit `--execute-gpu-training` flag.

## Exact remote-GPU commands

Run from a clean checkout of the branch/commit containing this directory:

```bash
python -m pip install psutil flash-linear-attention
python -m pip install -e . --no-build-isolation

python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
areno env --json
areno check

python examples/agentic/trainable_turns_ablation/run_ablation.py \
  --run-root artifacts/issue-199-ablation \
  --count 32 \
  --dataset-seed 2026 \
  --train-seed 2026 \
  --max-steps 10 \
  --prepare
```

Inspect `artifacts/issue-199-ablation/manifest.json` before authorizing GPU
training. The following is the single explicit command that runs all arms:

```bash
python examples/agentic/trainable_turns_ablation/run_ablation.py \
  --run-root artifacts/issue-199-ablation \
  --count 32 \
  --dataset-seed 2026 \
  --train-seed 2026 \
  --max-steps 10 \
  --execute-gpu-training
```

For auditability, these are the exact equivalent commands using repository-
relative paths. The manifest records the resolved absolute paths actually
executed:

```bash
areno train --ckpt Qwen/Qwen3-0.6B --model-hub modelscope \
  --dataset-path artifacts/issue-199-ablation/dataset/tictactoe.jsonl \
  --dataset-loader-fn examples/agentic/tictactoe/dataset_loader.py \
  --reward-fn-path examples/agentic/tictactoe/reward.py \
  --agent-fn examples/agentic/trainable_turns_ablation/run_agent.py \
  --algo gspo --seed 2026 --tp-size 1 --world-size 1 --batch-size 1 --n-samples 4 \
  --mini-bs 4 --max-running-prompts 4 --max-prompt-tokens 512 \
  --max-new-tokens 64 --max-context-len 2048 --agent-timeout-s 900 \
  --attn-backend native --max-steps 10 \
  --metrics-log-dir artifacts/issue-199-ablation/arms/all_assistant/metrics \
  --trainable-turns all_assistant

areno train --ckpt Qwen/Qwen3-0.6B --model-hub modelscope \
  --dataset-path artifacts/issue-199-ablation/dataset/tictactoe.jsonl \
  --dataset-loader-fn examples/agentic/tictactoe/dataset_loader.py \
  --reward-fn-path examples/agentic/tictactoe/reward.py \
  --agent-fn examples/agentic/trainable_turns_ablation/run_agent.py \
  --algo gspo --seed 2026 --tp-size 1 --world-size 1 --batch-size 1 --n-samples 4 \
  --mini-bs 4 --max-running-prompts 4 --max-prompt-tokens 512 \
  --max-new-tokens 64 --max-context-len 2048 --agent-timeout-s 900 \
  --attn-backend native --max-steps 10 \
  --metrics-log-dir artifacts/issue-199-ablation/arms/last_assistant/metrics \
  --trainable-turns last_assistant

areno train --ckpt Qwen/Qwen3-0.6B --model-hub modelscope \
  --dataset-path artifacts/issue-199-ablation/dataset/tictactoe.jsonl \
  --dataset-loader-fn examples/agentic/tictactoe/dataset_loader.py \
  --reward-fn-path examples/agentic/tictactoe/reward.py \
  --agent-fn examples/agentic/trainable_turns_ablation/run_agent.py \
  --algo gspo --seed 2026 --tp-size 1 --world-size 1 --batch-size 1 --n-samples 4 \
  --mini-bs 4 --max-running-prompts 4 --max-prompt-tokens 512 \
  --max-new-tokens 64 --max-context-len 2048 --agent-timeout-s 900 \
  --attn-backend native --max-steps 10 \
  --metrics-log-dir artifacts/issue-199-ablation/arms/final_answer/metrics \
  --trainable-turns final_answer
```

If training finishes but collection needs to be rerun:

```bash
python examples/agentic/trainable_turns_ablation/collect_metrics.py \
  --run-root artifacts/issue-199-ablation
```

Outputs:

- `manifest.json`: commit, dataset SHA-256, controlled difference, and commands
- `environment/*.json`: CUDA, `areno env --json`, and `areno check` outputs
- `ablation_steps.csv`: one row per arm and trainer step
- `ablation_steps.json`: the same rows with schema and manifest metadata

## Reproducibility boundary

Dataset generation, parent-process seeding, epoch order, and rollout request
seeds are deterministic. CUDA kernels may still prevent bitwise
reproducibility. Do not claim convergence differences from this ten-step smoke
ablation alone; use repeated runs in a separately approved extension.

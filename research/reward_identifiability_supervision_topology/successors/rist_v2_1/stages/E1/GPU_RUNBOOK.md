# E1 GPU runbook

Status: `AUTHORIZED_BUT_NOT_DEPLOYMENT_BOUND`

E1 starts only after C0 returns `PASS_C0_TO_FILTERED_TRAIN` and the structural
capacity dataset is written exactly as `data/train.jsonl`. Execute Qwen on a
verified 24GB-or-larger pairing and Gemma only on a verified 48GB-or-larger
pairing. The recommended Gemma resource is 80GB because 48GB is a canary floor,
not a prediction of 15% headroom.

For each family, follow the manifest mechanically:

1. Verify source/archive, authorization, model revision and weights, tokenizer,
   GPU UUID/total memory, driver/CUDA/PyTorch, extension import, empty GPU, and
   fresh output paths.
2. Start the monitored native serving command and send the frozen 32-task
   serving canary. Stop serving and retain monitor, journal, result, and log.
3. Run the monitored AF one-step command for GSPO, then its metrics and
   checkpoint-manifest commands. No retry or altered batch is permitted.
4. Load `step_000001`, send the one-task/four-turn reload canary, stop serving,
   and retain its journal/result/log. Repeat steps 3-4 for GRPO.
5. Assemble artifact-backed evidence and run `validate_capacity_evidence.py`.
   Reject the pairing for any OOM, nonzero command exit, zero/non-finite
   gradient, non-mixed reward group, missing artifact, failed reload, or peak
   memory above 85% of total.

Qwen and Gemma are separate decisions. Failure of one does not authorize
changing its model, batch, algorithm, task, seed, or GPU pairing inside the
consumed protocol.

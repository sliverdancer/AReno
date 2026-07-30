# GPU Handoff for the Frozen Q1 Pilot

Do not place passwords, private keys, access tokens, or ModelScope credentials
in this file or in chat. Configure secrets through SSH configuration, the
remote credential store, or a project-local ignored `.env`.

Provide the following non-secret information when the rental is ready:

```text
Host alias or access method:
Operating system:
GPU model:
GPU count:
VRAM per GPU:
CUDA/driver version, if known:
Python/PyTorch environment path, if preinstalled:
Available disk space:
Maximum rental duration:
Maximum approved spend or GPU-hours:
Outbound network / ModelScope access:
Repository checkout path, if already cloned:
```

## Read-only acceptance probe

Before any model download or training, the next agent must verify:

1. `nvidia-smi` reports the expected device and free memory;
2. PyTorch sees CUDA and records its versions;
3. disk space is sufficient for checkpoint, optimizer state, rollouts, and
   artifacts;
4. the repository checkout is on `research/trainable-turns-ablation`;
5. the source state matches the frozen local commit/diff artifact;
6. the qualification dataset hash matches `pilot_manifest.json`;
7. no existing Q1 metrics directory would be overwritten.

## Authorization boundary

GPU inspection is not training authorization. The frozen Q1 commands may run
only after explicit approval to execute GPU training on the accepted host.
Source changes remain local and version controlled; do not edit source directly
on a remote training host.


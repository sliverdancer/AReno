# AutoDL P3 handoff

Protocol: `CARE-P3-PILOT-v0.1`
Provider: AutoDL
Frozen target: one A800 80GB
GPU training authorization: not yet granted

## Information to send after renting

Send only non-secret instance metadata:

- GPU model and count;
- hourly price and billing mode;
- region;
- operating-system/image name;
- CPU core count and RAM;
- system/data-disk free space;
- CUDA driver version;
- public network availability;
- SSH host, port, and username only if remote access is required.

Do not paste passwords, private keys, API tokens, AutoDL developer tokens, or
account cookies into chat. Configure an SSH public key in AutoDL or keep secrets
in a local protected file outside the repository.

## Read-only inventory command

Run this after the instance starts and return the output:

```bash
python --version
git --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi
free -h
df -h
```

If PyTorch is already installed:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

These commands do not train or serve a model.

## Admission checks

Remote preparation stops unless all are true:

1. exactly one NVIDIA A800 with at least 80GB visible memory;
2. x86-64 Linux;
3. at least 8 logical CPU cores and 64GB RAM;
4. at least 60GB free disk;
5. live price at most CNY 7.50/hour and projected total at most CNY 60;
6. driver/toolkit combination can support PyTorch 2.6 or newer;
7. ModelScope and the Git remote are reachable;
8. the reviewed branch can be checked out at the exact frozen commit.

If the card differs from A800 but has at least 80GB memory, stop and re-freeze
the hardware manifest before proceeding. Do not silently substitute H800,
H20, PRO 6000, or a multi-GPU configuration.

## Remote sequence

After the instance passes inventory:

1. fetch and detach-checkout the reviewed commit;
2. confirm `git status --short` is empty;
3. install the environment following `AGENTS.md`;
4. run the scoped CPU test suite;
5. download `Qwen/Qwen3-0.6B` only from ModelScope;
6. verify every model file against `modelscope_asset.json`;
7. generate the P3 dataset and manifest outside the checkout;
8. run `execute_p3.py` without `--execute-gpu-training`;
9. report the preflight evidence and ask for explicit GPU-training approval;
10. only after approval, execute the six frozen runs and collect artifacts.

Renting or starting the instance does not by itself authorize step 10.

## Stop and billing rules

- Record the instance start time immediately.
- Stop at the first failed admission check.
- Stop the instance before eight billed hours under every outcome.
- Stop the pilot after the first timeout, non-zero exit, non-finite update,
  source drift, asset drift, or artifact-contract failure.
- Preserve failed logs; never delete and selectively rerun a bad seed.
- Shut down the instance after artifacts are downloaded and hashes recorded.

## Post-P3 route

If P3 passes, perform P4 on CPU using only the qualification artifacts:

- estimate runtime and variance;
- freeze confirmatory seed count and task count;
- decide whether five primary arms and at least five seeds fit the resource
  ceiling;
- return `KILL_UNDERPOWERED_OR_UNAFFORDABLE` if they do not.

No confirmatory P5 training opens automatically after a P3 pass.

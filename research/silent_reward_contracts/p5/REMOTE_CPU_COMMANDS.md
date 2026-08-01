# Exact P5 CPU reproduction commands

These commands require Git and Python 3.10+ only. They do not require CUDA,
PyTorch, model weights, serving, or paid APIs.

```bash
git clone --branch research/silent-reward-contracts \
  https://github.com/sliverdancer/AReno.git areno-arca
cd areno-arca

mkdir -p /tmp/arca-p5-upstreams
git clone --filter=blob:none --no-checkout https://github.com/THUDM/slime.git /tmp/arca-p5-upstreams/slime
git -C /tmp/arca-p5-upstreams/slime checkout --detach aaf5c2092b01219fa0d5c2d323741d409086ca32
git clone --filter=blob:none --no-checkout https://github.com/AgentR1/Agent-R1.git /tmp/arca-p5-upstreams/agent-r1
git -C /tmp/arca-p5-upstreams/agent-r1 checkout --detach b124aa46534cbf2fb8bc8af11405774984c42ac7
git clone --filter=blob:none --no-checkout https://github.com/mll-lab-nu/RAGEN.git /tmp/arca-p5-upstreams/ragen
git -C /tmp/arca-p5-upstreams/ragen checkout --detach 20daedc47558e000f7de912b060646bf2e8026bd
git clone --filter=blob:none --no-checkout https://github.com/rllm-org/rllm.git /tmp/arca-p5-upstreams/rllm
git -C /tmp/arca-p5-upstreams/rllm checkout --detach 75926c15e58fa29e4183d01292d10462d2047be9
git clone --filter=blob:none --no-checkout https://github.com/microsoft/agent-lightning.git /tmp/arca-p5-upstreams/agent-lightning
git -C /tmp/arca-p5-upstreams/agent-lightning checkout --detach f0a77cfad71e6222a3edb7dfc7a0f611bd231364

python3 -m research.silent_reward_contracts.p5.external_validation \
  --upstreams-root /tmp/arca-p5-upstreams

python3 -m venv /tmp/arca-p5-pytest
/tmp/arca-p5-pytest/bin/python -m pip install pytest
/tmp/arca-p5-pytest/bin/python -m pytest -q tests/test_arca_p5_cpu.py

sha256sum research/silent_reward_contracts/p5/artifacts/*
```

The generator verifies every Git HEAD and audited file hash before executing
the extracted upstream function bodies. A mismatch terminates artifact
generation rather than accepting source drift.

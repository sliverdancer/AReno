# Exact P2 CPU reproduction

No model weights or GPU are needed.

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/volcengine/verl.git /tmp/arca-upstreams/verl
git -C /tmp/arca-upstreams/verl checkout --detach \
  e9618406de5bad40041d7612554e465ec2003ec1

python -m research.silent_reward_contracts.generate_p2_artifacts \
  --verl-root /tmp/arca-upstreams/verl \
  --output-dir research/silent_reward_contracts/p2/artifacts

python -m pytest tests/test_silent_reward_contracts_cpu.py -q
sha256sum research/silent_reward_contracts/p2/artifacts/*
```

Before accepting the result, compare the source hashes in
`upstream_manifest.json` with the cloned checkout. A mismatch invalidates the
artifact; do not silently continue against another commit.

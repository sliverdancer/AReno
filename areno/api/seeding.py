"""Deterministic seed helpers shared by trainer implementations."""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Sequence
from typing import Any


def seed_parent_process(seed: int) -> None:
    """Seed parent-process RNGs before backend/model initialization."""

    normalized = _normalize_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(normalized)
    random.seed(normalized)
    try:
        import numpy as np

        np.random.seed(normalized % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(normalized)
    except ImportError:
        pass


def derive_seed(seed: int, *parts: Any) -> int:
    """Derive a stable non-negative 63-bit seed from a base seed and labels."""

    normalized = _normalize_seed(seed)
    payload = json_seed_material(normalized, parts)
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def epoch_dataset_view(dataset, *, seed: int, epoch: int):
    """Return a deterministic shuffled view without copying dataset records."""

    indices = list(range(len(dataset)))
    random.Random(derive_seed(seed, "dataset", epoch)).shuffle(indices)
    return _IndexedDatasetView(dataset, indices, seed=derive_seed(seed, "dataset", epoch), epoch=epoch)


def json_seed_material(seed: int, parts: Sequence[Any]) -> bytes:
    """Encode seed derivation inputs without relying on process hash state."""

    fields = [str(seed), *(f"{type(part).__name__}:{part}" for part in parts)]
    return "\x1f".join(fields).encode("utf-8")


class _IndexedDatasetView:
    """Minimal dataset-like view used by all trainer loops."""

    def __init__(self, dataset, indices: list[int], *, seed: int, epoch: int):
        self._dataset = dataset
        self._indices = indices
        self.seed = seed
        self.epoch = epoch

    @property
    def order_sha256(self) -> str:
        payload = ",".join(str(index) for index in self._indices).encode()
        return hashlib.sha256(payload).hexdigest()

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int):
        return self._dataset[self._indices[index]]


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed

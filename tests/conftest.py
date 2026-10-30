"""Shared fixtures: tiny deterministic models and random vector helpers."""

from __future__ import annotations

import pytest
import torch

from tmem.models import tiny_causal, tiny_encdec


@pytest.fixture(scope="session")
def causal_bundle():
    return tiny_causal(d_model=64, layers=2, seed=7)


@pytest.fixture(scope="session")
def encdec_bundle():
    return tiny_encdec(d_model=64, layers=2, seed=7)


@pytest.fixture()
def rand_vecs():
    """Deterministic pseudo-random vectors of shape (n, d)."""

    def make(n: int = 4, d: int = 64, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(n, d, generator=g)

    return make

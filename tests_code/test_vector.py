"""Offline unit tests for codemem.vector (cosine, centroid, centering, MMR)."""

from __future__ import annotations

import torch

from codemem.vector import RunningCentroid, cosine_sim, mmr, normalize, pairwise_cosine


def test_normalize_zero_row_stays_zero():
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    out = normalize(x)
    assert torch.all(out[0] == 0)
    assert torch.allclose(out[1].norm(), torch.tensor(1.0))
    assert not torch.isnan(out).any()


def test_cosine_sim_basic():
    q = torch.tensor([1.0, 0.0])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sim = cosine_sim(q, keys)
    assert torch.allclose(sim, torch.tensor([1.0, 0.0, -1.0]), atol=1e-6)


def test_cosine_sim_empty_keys():
    q = torch.tensor([1.0, 0.0])
    assert cosine_sim(q, torch.zeros((0, 2))).shape == (0,)


def test_pairwise_cosine_shape_and_diagonal():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    m = pairwise_cosine(x, x)
    assert m.shape == (2, 2)
    assert torch.allclose(m.diag(), torch.ones(2), atol=1e-6)


def test_running_centroid_mean_and_center():
    c = RunningCentroid()
    assert c.value is None
    v = torch.tensor([2.0, 4.0])
    assert torch.equal(c.center(v), v)          # no updates -> unchanged
    c.update(torch.tensor([1.0, 2.0]))
    c.update(torch.tensor([3.0, 6.0]))
    assert torch.allclose(c.value, torch.tensor([2.0, 4.0]))
    assert len(c) == 2
    centered = c.center(torch.tensor([[3.0, 6.0], [1.0, 2.0]]))
    assert torch.allclose(centered, torch.tensor([[1.0, 2.0], [-1.0, -2.0]]))


def test_mmr_pure_relevance_is_topk_cosine():
    q = torch.tensor([1.0, 0.0])
    keys = torch.tensor([[0.5, 0.866], [0.9, 0.436], [0.0, 1.0]])
    assert mmr(q, keys, k=2, lambda_=1.0) == [1, 0]


def test_mmr_diversifies_clone_flood():
    # three identical high-relevance clones + one slightly-less-relevant but
    # dissimilar candidate; lambda=0.5 must pick the diverse one second.
    q = torch.tensor([1.0, 0.0])
    clone = torch.tensor([0.9, 0.436])
    distinct = torch.tensor([0.85, -0.527])
    keys = torch.stack([clone, clone, clone, distinct])
    sel = mmr(q, keys, k=2, lambda_=0.5)
    assert sel[0] == 0          # most relevant first
    assert sel[1] == 3          # diverse candidate beats a clone
    # pure relevance would instead take two clones
    assert mmr(q, keys, k=2, lambda_=1.0) == [0, 1]


def test_mmr_empty_and_nonpositive_k():
    q = torch.tensor([1.0, 0.0])
    assert mmr(q, torch.zeros((0, 2)), k=3) == []
    assert mmr(q, torch.eye(2), k=0) == []


def test_mmr_deterministic_tie_break():
    q = torch.tensor([1.0, 0.0])
    keys = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert mmr(q, keys, k=2, lambda_=1.0) == [0, 1]
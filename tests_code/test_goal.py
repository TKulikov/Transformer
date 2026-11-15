"""Offline tests for goal decomposition (pure) and dendrogram root_split."""

from __future__ import annotations

import torch

from codemem.cluster import dendrogram, root_split
from codemem.goal import decompose


def _pairs():
    # two tight pairs, far apart -> clean root split
    return [
        ("a1", torch.tensor([1.0, 0.0])),
        ("a2", torch.tensor([0.99, 0.14])),
        ("b1", torch.tensor([0.0, 1.0])),
        ("b2", torch.tensor([0.14, 0.99])),
    ]


def test_root_split_groups_and_gap():
    items = _pairs()
    embs = torch.stack([e for _, e in items])
    ga, gb, gap = root_split(dendrogram(embs), 4)
    assert {0, 1} in (set(ga), set(gb))
    assert {2, 3} in (set(ga), set(gb))
    assert gap > 0.5


def test_decompose_accepts_clean_split():
    nodes = decompose(_pairs(), tau_split=0.15, judge=lambda a, b: True, depth_cap=2)
    assert len(nodes) == 2
    assert all(not n.atomic for n in nodes)
    assert all(len(n.children) == 2 for n in nodes)
    assert all(c.atomic for n in nodes for c in n.children)


def test_decompose_rejected_judge_stays_atomic():
    nodes = decompose(_pairs(), tau_split=0.15, judge=lambda a, b: False, depth_cap=2)
    assert len(nodes) == 4
    assert all(n.atomic for n in nodes)


def test_decompose_depth_cap_zero_atomic():
    nodes = decompose(_pairs(), tau_split=0.15, judge=lambda a, b: True, depth_cap=0)
    assert len(nodes) == 4 and all(n.atomic for n in nodes)


def test_decompose_no_gap_atomic():
    items = [("x", torch.tensor([1.0, 0.0])), ("y", torch.tensor([0.99, 0.14]))]
    nodes = decompose(items, tau_split=0.5, judge=lambda a, b: True, depth_cap=2)
    assert all(n.atomic for n in nodes)        # gap below tau -> no split
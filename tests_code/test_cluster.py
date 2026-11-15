"""Offline unit tests for codemem.cluster (dendrogram + largest-gap cut)."""

from __future__ import annotations

import torch

from codemem.cluster import dendrogram, largest_gap_cut


def test_dendrogram_trivial():
    assert dendrogram(torch.zeros((0, 2))) == []
    assert dendrogram(torch.tensor([[1.0, 0.0]])) == []


def test_dendrogram_merge_count_and_root():
    vecs = torch.tensor([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]])
    merges = dendrogram(vecs)
    assert len(merges) == 2                     # n-1 merges
    assert merges[-1].size == 3                 # root holds all leaves
    assert set(merges[-1].members) == {0, 1, 2}
    assert merges[0].height <= merges[1].height  # increasing height


def test_largest_gap_cut_separates_tight_pairs():
    vecs = torch.tensor([
        [1.0, 0.0], [0.99, 0.14],               # tight pair A
        [0.0, 1.0], [0.14, 0.99],               # tight pair B
    ])
    merges = dendrogram(vecs)
    clusters, gap = largest_gap_cut(merges, n_leaves=4)
    assert len(clusters) == 2
    assert gap > 0.5                            # big jump between within/between pair
    flat = sorted(i for c in clusters for i in c)
    assert flat == [0, 1, 2, 3]                 # partition covers all leaves
    assert {0, 1} in [set(c) for c in clusters]
    assert {2, 3} in [set(c) for c in clusters]


def test_largest_gap_cut_no_split_possible():
    clusters, gap = largest_gap_cut([], n_leaves=1)
    assert clusters == [(0,)]
    assert gap == 0.0
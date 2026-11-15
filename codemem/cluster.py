"""Agglomerative clustering for goal dendrograms (average linkage, pure torch).

scipy is not a project dependency, so the small dendrogram used to decide
whether a goal splits into subgoals is implemented here. Goal sets are tiny
(2-8 candidates), so the O(n^3) recomputation is negligible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .vector import pairwise_cosine


@dataclass(frozen=True)
class Merge:
    """One agglomeration step.

    Attributes:
        a, b: merged cluster ids. Leaves are 0..n-1; internal nodes n..2n-2.
        height: merge distance = 1 - mean pairwise cosine between the two
            clusters' members (average linkage). Larger = less similar.
        size: number of leaves in the new cluster.
        members: tuple of leaf indices in the new cluster.
    """

    a: int
    b: int
    height: float
    size: int
    members: tuple[int, ...]


def dendrogram(vecs: torch.Tensor) -> list[Merge]:
    """Average-linkage agglomerative clustering of rows of vecs [N, H].

    Distance between two clusters is the mean of (1 - cosine) over all member
    pairs. Returns the merge history in increasing height order; the last merge
    builds the root. N <= 1 returns [] (nothing to merge). Deterministic: ties
    merge the lowest-index pair first.
    """
    n = vecs.shape[0]
    if n <= 1:
        return []
    dist = 1.0 - pairwise_cosine(vecs, vecs)   # [N, N], zero diagonal
    clusters: dict[int, tuple[int, ...]] = {i: (i,) for i in range(n)}
    merges: list[Merge] = []
    next_id = n
    while len(clusters) > 1:
        ids = sorted(clusters)
        best = None
        best_h = float("inf")
        for ai in range(len(ids)):
            for bi in range(ai + 1, len(ids)):
                a, b = ids[ai], ids[bi]
                ma, mb = clusters[a], clusters[b]
                h = float(dist[list(ma)][:, list(mb)].mean())
                if h < best_h:
                    best_h, best = h, (a, b)
        a, b = best
        members = tuple(sorted(clusters[a] + clusters[b]))
        merges.append(Merge(a=a, b=b, height=best_h, size=len(members), members=members))
        del clusters[a]
        del clusters[b]
        clusters[next_id] = members
        next_id += 1
    return merges


def largest_gap_cut(merges: list[Merge], n_leaves: int) -> tuple[list[tuple[int, ...]], float]:
    """Cut the dendrogram at the largest gap between consecutive merge heights.

    The gap heuristic: a big jump in merge height separates tight sub-clusters
    from the loose glue above them, so cutting just below the jump yields the
    natural clusters. Used to propose a goal split before the self-sufficiency
    judge validates it.

    Args:
        merges: merge history from :func:`dendrogram` (increasing height).
        n_leaves: number of original items.

    Returns:
        (clusters, gap): clusters is a list of leaf-index tuples partitioning
        0..n_leaves-1; gap is the height jump at the cut (0.0 when no split is
        possible). With < 2 merges or n_leaves <= 1, returns one cluster holding
        all leaves and gap 0.0.
    """
    if n_leaves <= 1 or len(merges) < 2:
        return [tuple(range(n_leaves))], 0.0
    heights = [m.height for m in merges]
    gaps = [heights[i + 1] - heights[i] for i in range(len(heights) - 1)]
    cut = max(range(len(gaps)), key=lambda i: gaps[i])
    gap = gaps[cut]
    # clusters present just before merge index (cut + 1) fires
    clusters: dict[int, tuple[int, ...]] = {i: (i,) for i in range(n_leaves)}
    next_id = n_leaves
    for m in merges[: cut + 1]:
        del clusters[m.a]
        del clusters[m.b]
        clusters[next_id] = m.members
        next_id += 1
    return [clusters[i] for i in sorted(clusters)], gap
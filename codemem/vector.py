"""Vector math for codemem: cosine, running centroid, centering, MMR.

Pure functions over float32 torch tensors on CPU. No model dependency, so the
whole module is unit-testable offline. Embeddings are row vectors: a single
vector has shape [H], a batch has shape [N, H].
"""

from __future__ import annotations

import torch


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize along the last dimension.

    Zero rows stay zero (the norm is clamped to ``eps``), so the result never
    contains NaN/Inf. Works for [H] and [N, H] inputs alike.
    """
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def cosine_sim(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of one query [H] against N keys [N, H] -> [N].

    Empty keys ([0, H]) return an empty [0] tensor. Zero vectors yield 0.
    """
    if keys.numel() == 0:
        return keys.new_zeros((0,))
    q = normalize(query.float())
    k = normalize(keys.float())
    return k @ q


def pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity matrix between rows of a [N, H] and b [M, H] -> [N, M]."""
    return normalize(a.float()) @ normalize(b.float()).T


class RunningCentroid:
    """Incremental mean of inserted vectors — the "living topic" of a node.

    Keys themselves stay static; centering subtracts this evolving mean at query
    time, so the centroid is the only moving part (static storage + dynamic
    selection). Lifecycle: call :meth:`update` per inserted key, read
    :attr:`value` / :meth:`center` at any time.

    With zero updates, :attr:`value` is None and :meth:`center` returns the input
    unchanged (nothing to subtract yet). Not thread-safe.
    """

    def __init__(self) -> None:
        self._sum: torch.Tensor | None = None
        self._n: int = 0

    def update(self, vec: torch.Tensor) -> None:
        """Accumulate one vector [H] into the running mean. Mutates self only."""
        v = vec.detach().float().cpu()
        self._sum = v.clone() if self._sum is None else self._sum + v
        self._n += 1

    @property
    def value(self) -> torch.Tensor | None:
        """Current mean [H], or None before the first update."""
        if self._n == 0 or self._sum is None:
            return None
        return self._sum / self._n

    def center(self, vecs: torch.Tensor) -> torch.Tensor:
        """Subtract the running mean from vecs ([H] or [N, H]).

        Returns vecs unchanged (as float32) when no updates have happened.
        """
        v = vecs.detach().float().cpu()
        mean = self.value
        return v if mean is None else v - mean

    def __len__(self) -> int:
        return self._n


def mmr(
    query: torch.Tensor,
    keys: torch.Tensor,
    k: int,
    lambda_: float = 0.5,
    relevance: torch.Tensor | None = None,
) -> list[int]:
    """Maximal Marginal Relevance selection over keys [N, H] for query [H].

    Greedily picks indices maximizing
    ``lambda_ * rel(c) - (1 - lambda_) * max_{s selected} sim(c, s)``.
    This is the diversification workhorse: it breaks clone-dominated rankings
    (the robust never-worse mechanism found in the dialogue experiments).

    Args:
        query: [H] query vector.
        keys: [N, H] candidate vectors.
        k: number of indices to select; clamped to [0, N].
        lambda_: relevance/diversity trade-off in [0, 1]. 1 = pure top-k by
            relevance, 0 = pure diversity.
        relevance: optional [N] relevance scores overriding cosine(query, keys)
            (e.g. cosine + a maturity bonus). Diversity still uses key-key
            cosine. None means plain cosine relevance.

    Returns:
        Selected key indices in pick order (length min(k, N)). Empty keys or
        k <= 0 give []. Ties break toward the lowest index (deterministic).
    """
    n = keys.shape[0]
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    kn = normalize(keys.float())
    qn = normalize(query.float())
    rel = kn @ qn if relevance is None else relevance.float()   # [N]
    sim = kn @ kn.T                     # [N, N] inter-candidate similarity
    selected: list[int] = []
    remaining = list(range(n))
    max_to_sel = torch.full((n,), -float("inf"))
    for _ in range(k):
        best_i, best_score = -1, -float("inf")
        for i in remaining:
            penalty = 0.0 if not selected else float(max_to_sel[i])
            score = lambda_ * float(rel[i]) - (1.0 - lambda_) * penalty
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        remaining.remove(best_i)
        # update max similarity to the newly selected candidate
        max_to_sel = torch.maximum(max_to_sel, sim[best_i])
    return selected
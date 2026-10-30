"""KV-cache policies: classic (unbounded) and sparse (budgeted) caches.

The cache itself is always the native ``transformers`` ``DynamicCache``
(``classic`` mode never touches it); the sparse policy is implemented as a
post-step *pruning* of the cache:

* ``sliding`` — StreamingLLM-style: keep the first ``sink`` tokens and the
  most recent ones. Remaining positions stay contiguous, so relative
  position encodings (T5) stay consistent.
* ``topk``     — H2O-style: keep the sink tokens, the recent window and the
  tokens with the highest accumulated attention scores in between. Requires
  attention weights (models must run with the eager attention
  implementation); absolute positions of kept keys are already baked into
  cached K vectors (RoPE), so pruning is position-safe for causal models.

Both policies keep the token produced in the current step visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

CacheMode = Literal["classic", "sparse"]
EvictStrategy = Literal["sliding", "topk"]


@dataclass(frozen=True)
class KVConfig:
    """Budgeted KV-cache configuration.

    Args:
        mode: ``classic`` = unbounded cache, ``sparse`` = prune to budget.
        strategy: which tokens survive the pruning.
        budget: maximal number of cached token slots (sparse only).
        sink: number of initial tokens that are never evicted.
        recent: number of trailing tokens always kept.
    """

    mode: CacheMode = "classic"
    strategy: EvictStrategy = "sliding"
    budget: int = 256
    sink: int = 4
    recent: int = 64

    def __post_init__(self) -> None:
        if self.mode not in ("classic", "sparse"):
            raise ValueError(f"unknown cache mode: {self.mode!r}")
        if self.strategy not in ("sliding", "topk"):
            raise ValueError(f"unknown eviction strategy: {self.strategy!r}")
        if self.mode == "sparse" and self.budget < self.sink + 2:
            raise ValueError("budget must keep at least the sink tokens and 2 recent ones")

    @property
    def needs_scores(self) -> bool:
        return self.mode == "sparse" and self.strategy == "topk"

    @classmethod
    def classic(cls) -> "KVConfig":
        return cls(mode="classic")

    @classmethod
    def sparse(cls, budget: int, strategy: EvictStrategy = "sliding", sink: int = 4, recent: int | None = None) -> "KVConfig":
        recent = recent if recent is not None else max(budget - sink, 1)
        return cls(mode="sparse", strategy=strategy, budget=budget, sink=sink, recent=recent)


def keep_indices(length: int, cfg: KVConfig, importance: torch.Tensor | None = None) -> torch.Tensor:
    """Indices (ascending) of cache slots that survive one pruning pass.

    Args:
        length: current number of cached slots.
        cfg: cache policy.
        importance: per-slot scores of shape ``(length,)`` for ``topk``.
    """
    if cfg.mode == "classic" or length <= cfg.budget:
        return torch.arange(length)
    keep_sink = min(cfg.sink, cfg.budget - 1)
    keep_tail = min(cfg.recent, cfg.budget - keep_sink)
    tail_start = length - keep_tail
    head = torch.arange(keep_sink)
    tail = torch.arange(tail_start, length)

    if cfg.strategy == "sliding":
        # Everything between sink and recent is dropped, oldest first:
        # equivalently keep sink tokens plus the last (budget - sink) slots.
        n_tail = max(cfg.budget - keep_sink, 1)
        tail = torch.arange(max(length - n_tail, keep_sink), length)
        return torch.cat([head, tail]).unique()

    # topk: rank the middle region by accumulated attention importance.
    middle = torch.arange(keep_sink, tail_start)
    n_mid = cfg.budget - keep_sink - keep_tail
    if n_mid <= 0 or middle.numel() == 0:
        return torch.cat([head, tail]).unique()
    if importance is None or importance.numel() < length:
        raise ValueError("topk eviction requires per-slot importance scores")
    mid_scores = importance.to(middle.device)[middle]
    _, order = torch.topk(mid_scores, k=min(n_mid, middle.numel()))
    selected = middle[order.sort().values]
    return torch.cat([head, selected, tail]).unique()


class ImportanceTracker:
    """Accumulates H2O-style per-slot attention mass from attention weights."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.scores = torch.zeros(0, device=self.device)

    def step(self, attentions: tuple[torch.Tensor | None, ...] | None) -> None:
        """Add one decoding step's attention mass: ``a`` is ``(1, heads, q, k)``."""
        if not attentions:
            return
        cols = []
        for a in attentions:
            if a is None:
                continue
            cols.append(a.detach().float().sum(dim=(0, 1, 2)))  # (k,)
        if not cols:
            return
        mass = torch.stack(cols).mean(dim=0).to(self.device)
        if self.scores.numel() != mass.numel():
            pad = torch.zeros(mass.numel() - self.scores.numel(), device=self.device)
            self.scores = torch.cat([self.scores, pad])
        self.scores += mass

    def prune(self, keep: torch.Tensor) -> None:
        """Renumber slots after the cache was pruned to ``keep``."""
        self.scores = self.scores.to(keep.device)[keep].clone().to(self.device)

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
EvictStrategy = Literal["sliding", "topk", "snapkv", "pyramidkv"]


@dataclass(frozen=True)
class KVConfig:
    """Configure unbounded, dynamic-eviction, or prompt-compressing KV caches.

    ``budget`` is the maximum slots per layer, except for PyramidKV where it
    is the mean across layers. ``sink``/``recent`` control sliding and H2O;
    ``observation``/``kernel_size`` control SnapKV-style prompt selection;
    ``pyramid_beta`` controls the decreasing layer budgets of PyramidKV.
    ``chunk_size`` bounds attention-score materialisation during H2O prefill.
    Invalid modes, strategies, non-positive windows, or budgets that cannot
    hold the mandatory sink/recent tokens raise ``ValueError``.
    """

    mode: CacheMode = "classic"
    strategy: EvictStrategy = "sliding"
    budget: int = 256
    sink: int = 4
    recent: int = 64
    observation: int = 64
    kernel_size: int = 7
    pyramid_beta: float = 20.0
    chunk_size: int = 256

    def __post_init__(self) -> None:
        if self.mode not in ("classic", "sparse"):
            raise ValueError(f"unknown cache mode: {self.mode!r}")
        if self.strategy not in ("sliding", "topk", "snapkv", "pyramidkv"):
            raise ValueError(f"unknown eviction strategy: {self.strategy!r}")
        if self.mode == "sparse" and self.budget < self.sink + 2:
            raise ValueError("budget must keep at least the sink tokens and 2 recent ones")
        if min(self.observation, self.kernel_size, self.chunk_size) <= 0 or self.pyramid_beta <= 0:
            raise ValueError("observation, kernel_size, chunk_size and pyramid_beta must be positive")

    @property
    def needs_scores(self) -> bool:
        """Whether any decoding phase must materialise attention weights."""
        return self.mode == "sparse" and self.strategy in ("topk", "snapkv", "pyramidkv")

    @property
    def needs_step_scores(self) -> bool:
        """Whether every generation step needs attention weights (H2O only)."""
        return self.mode == "sparse" and self.strategy == "topk"

    @property
    def compresses_prompt(self) -> bool:
        """Whether the policy selects prompt K/V once before generation."""
        return self.mode == "sparse" and self.strategy in ("snapkv", "pyramidkv")

    @classmethod
    def classic(cls) -> "KVConfig":
        return cls(mode="classic")

    @classmethod
    def sparse(
        cls,
        budget: int,
        strategy: EvictStrategy = "sliding",
        sink: int = 4,
        recent: int | None = None,
        *,
        observation: int = 64,
        kernel_size: int = 7,
        pyramid_beta: float = 20.0,
        chunk_size: int = 256,
    ) -> "KVConfig":
        """Create a sparse policy; omitted method parameters use paper-style defaults."""
        recent = recent if recent is not None else max(budget - sink, 1)
        return cls(
            mode="sparse", strategy=strategy, budget=budget, sink=sink, recent=recent,
            observation=observation, kernel_size=kernel_size, pyramid_beta=pyramid_beta,
            chunk_size=chunk_size,
        )


def keep_indices(length: int, cfg: KVConfig, importance: torch.Tensor | None = None) -> torch.Tensor:
    """Indices (ascending) of cache slots that survive one pruning pass.

    Args:
        length: current number of cached slots.
        cfg: cache policy.
        importance: per-slot scores of shape ``(length,)`` for ``topk``.
    """
    device = importance.device if importance is not None else torch.device("cpu")
    if cfg.mode == "classic" or length <= cfg.budget:
        return torch.arange(length, device=device)
    keep_sink = min(cfg.sink, cfg.budget - 1)
    keep_tail = min(cfg.recent, cfg.budget - keep_sink)
    tail_start = length - keep_tail
    head = torch.arange(keep_sink, device=device)
    tail = torch.arange(tail_start, length, device=device)

    if cfg.strategy == "sliding":
        # Everything between sink and recent is dropped, oldest first:
        # equivalently keep sink tokens plus the last (budget - sink) slots.
        n_tail = max(cfg.budget - keep_sink, 1)
        tail = torch.arange(max(length - n_tail, keep_sink), length, device=device)
        return torch.cat([head, tail]).unique()

    # topk: rank the middle region by accumulated attention importance.
    middle = torch.arange(keep_sink, tail_start, device=device)
    n_mid = cfg.budget - keep_sink - keep_tail
    if n_mid <= 0 or middle.numel() == 0:
        return torch.cat([head, tail]).unique()
    if importance is None or importance.numel() < length:
        raise ValueError("topk eviction requires per-slot importance scores")
    mid_scores = importance.to(middle.device)[middle]
    _, order = torch.topk(mid_scores, k=min(n_mid, middle.numel()))
    selected = middle[order.sort().values]
    return torch.cat([head, selected, tail]).unique()


def snapkv_indices(
    attention: torch.Tensor,
    budget: int,
    observation: int = 64,
    kernel_size: int = 7,
    num_key_value_heads: int | None = None,
) -> torch.Tensor:
    """Select prompt positions independently for each K/V head.

    ``attention`` contains one batch of normalized weights with shape
    ``(1, query_heads, query_length, key_length)``; only its last
    ``observation`` queries vote. Query heads that share a K/V head are summed
    before max-pooling and top-k selection. The complete observation tail is
    retained. Returns sorted indices with shape ``(kv_heads, budget)`` when
    compression occurs, or ``(kv_heads, key_length)`` when the prompt already
    fits. Non-batch inputs and a budget smaller than the observation window
    raise ``ValueError``; a budget equal to it keeps only that window.
    """
    if attention.ndim != 4 or attention.shape[0] != 1:
        raise ValueError("attention must have shape (1, query_heads, queries, keys)")
    if budget < observation:
        raise ValueError("budget must be at least observation")
    query_heads, key_length = attention.shape[1], attention.shape[-1]
    kv_heads = num_key_value_heads or query_heads
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by num_key_value_heads")
    if key_length <= budget:
        return torch.arange(key_length, device=attention.device).expand(kv_heads, -1)
    groups = query_heads // kv_heads
    votes = attention[0, :, -observation:, :].float().reshape(kv_heads, groups, observation, key_length).sum(dim=(1, 2))
    prefix, tail = votes[..., :-observation], votes[..., -observation:]
    if kernel_size > 1:
        prefix = torch.nn.functional.max_pool1d(
            prefix.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        ).squeeze(1)[..., : prefix.shape[-1]]
    selected = prefix.topk(budget - observation, dim=-1).indices.sort(dim=-1).values
    tail_indices = torch.arange(key_length - observation, key_length, device=attention.device)
    return torch.cat([selected, tail_indices.expand(kv_heads, -1)], dim=-1)


def pyramid_budgets(num_layers: int, average_budget: int, beta: float = 20.0, observation: int = 64) -> list[int]:
    """Allocate a decreasing arithmetic sequence of per-layer KV budgets.

    The returned positive integers sum exactly to ``num_layers *
    average_budget``. The top layer receives approximately
    ``average_budget / beta`` but never fewer than ``observation`` slots.
    ``num_layers`` and ``average_budget`` must be positive, and the bottom
    budget must exceed ``observation``; violations raise ``ValueError``.
    """
    if num_layers <= 0 or average_budget <= 0 or beta <= 0 or observation <= 0:
        raise ValueError("num_layers, average_budget, beta and observation must be positive")
    top = max(observation, round(average_budget / beta))
    bottom = 2 * average_budget - top
    if bottom <= observation:
        raise ValueError("average_budget is too small for the observation window")
    raw = torch.linspace(bottom, top, num_layers)
    budgets = raw.floor().to(torch.int64)
    remainder = num_layers * average_budget - int(budgets.sum())
    fractions = raw - budgets
    order = fractions.argsort(descending=True, stable=True)
    budgets[order[:remainder]] += 1
    return budgets.tolist()


def h2o_head_indices(scores: torch.Tensor, cfg: KVConfig, num_key_value_heads: int | None = None) -> torch.Tensor:
    """Apply accumulated-attention H2O eviction independently to K/V heads.

    ``scores`` has shape ``(query_heads, slots)`` and contains detached
    accumulated attention mass. Scores are summed within each grouped-query
    K/V group. The returned ``(kv_heads, retained_slots)`` indices preserve
    sinks, recent tokens, and the highest-scoring middle tokens.
    """
    if scores.ndim != 2:
        raise ValueError("scores must have shape (query_heads, slots)")
    query_heads, length = scores.shape
    kv_heads = num_key_value_heads or query_heads
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by num_key_value_heads")
    grouped = scores.float().reshape(kv_heads, query_heads // kv_heads, length).sum(dim=1)
    return torch.stack([keep_indices(length, cfg, grouped[h]) for h in range(kv_heads)])


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


class HeadImportanceTracker:
    """Accumulate per-layer, per-query-head attention mass for H2O.

    The tracker owns detached float32 score tensors and does not retain
    attention matrices. ``step`` accepts one batch of per-layer attention
    weights, growing new slots with zero history. ``prune`` reorders and
    truncates scores with the per-head indices returned by
    :func:`h2o_head_indices`. The class is stateful and not thread-safe.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.scores: list[torch.Tensor] = []

    def step(self, attentions: tuple[torch.Tensor | None, ...] | None) -> None:
        """Accumulate a forward pass; ``None`` layers leave existing scores unchanged."""
        if not attentions:
            return
        for layer, attention in enumerate(attentions):
            if attention is None:
                continue
            mass = attention.detach().float().sum(dim=(0, 2)).to(self.device)
            if layer == len(self.scores):
                self.scores.append(torch.zeros_like(mass))
            current = self.scores[layer]
            if current.shape != mass.shape:
                if current.shape[0] != mass.shape[0] or current.shape[1] > mass.shape[1]:
                    raise ValueError("attention score shape is incompatible with accumulated H2O state")
                current = torch.cat([current, torch.zeros_like(mass[:, current.shape[1] :])], dim=1)
            self.scores[layer] = current + mass

    def prune(self, keep: list[torch.Tensor]) -> None:
        """Apply one ``(kv_heads, retained)`` index tensor per scored layer."""
        if len(keep) != len(self.scores):
            raise ValueError("one keep tensor is required per scored layer")
        pruned = []
        for score, indices in zip(self.scores, keep):
            groups = score.shape[0] // indices.shape[0]
            expanded = indices.repeat_interleave(groups, dim=0)
            pruned.append(torch.gather(score, 1, expanded).clone())
        self.scores = pruned

"""External associative memory with TTL-based eviction.

Faithful rewrite of the ``Memory`` / ``CorrectionLayer`` classes from the
original notebook, with the following fixes:

* the buggy ``__empty_slots`` list (which could accumulate duplicate
  indices) is replaced by a boolean ``occupied`` mask;
* tensors live on an explicit device/dtype and migrate automatically when
  the model runs on another device (e.g. GPU);
* the unused ``end`` attribute was dropped.

Algorithm (unchanged):

* :meth:`Memory.get_relevant` retrieves the ``count`` most similar stored
  vectors to the query (cosine similarity, max over query positions),
  refreshes their TTL, decays every TTL by 1 and evicts expired slots.
* :meth:`Memory.update` merges a new vector into an existing slot when the
  similarity is high (>= ``crit``), stores it into a free slot when the
  similarity to memory is low (<= ``max_to_new``) *and* the vector passes
  the activity gate :meth:`Memory.check`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_EPS = 1e-8
_NORM_EPS = 1e-6


def _unit(x: torch.Tensor, dim: int) -> torch.Tensor:
    """L2-normalise ``x`` along ``dim``, keeping zero rows at zero."""
    norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)
    return x / torch.clamp(norm, min=_EPS)


class Memory:
    """Fixed-size slot memory (``size`` x ``d_model``) with cosine retrieval."""

    TTL_MAX = 6
    TTL_HIT_BONUS = 2
    TTL_NEW = 5

    def __init__(
        self,
        d_model: int,
        size: int = 512,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        merge_k: float | None = None,
    ) -> None:
        if d_model <= 0 or size <= 0:
            raise ValueError("d_model and size must be positive")
        if merge_k is not None and not 0.0 < merge_k <= 1.0:
            raise ValueError("merge_k must be in (0, 1]")
        self.size = size
        self.d_model = d_model
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.merge_k = merge_k
        self.clear()

    # ------------------------------------------------------------------ state
    def clear(self) -> None:
        """Reset the memory to its initial (empty) state."""
        self.context = torch.zeros(1, self.size, self.d_model, device=self.device, dtype=self.dtype)
        self.ttl = torch.zeros(self.size, device=self.device, dtype=self.dtype)
        self.occupied = torch.zeros(self.size, dtype=torch.bool, device=self.device)
        self.count_element = 0
        self.stats = {"get_calls": 0, "update_calls": 0, "retrievals": 0,
                      "stores": 0, "merges": 0, "evictions": 0, "gate_rejects": 0}

    def _to(self, device: torch.device, dtype: torch.dtype) -> None:
        """Migrate all state tensors to ``device``/``dtype`` if needed."""
        if self.context.device != device or self.context.dtype != dtype:
            self.context = self.context.to(device=device, dtype=dtype)
            self.ttl = self.ttl.to(device=device)
            self.occupied = self.occupied.to(device=device)
            self.device = device

    # -------------------------------------------------------------- retrieval
    def get_relevant(self, query: torch.Tensor, count: int = 256) -> tuple[torch.Tensor, int]:
        """Return up to ``count`` relevant stored vectors and refresh their TTL.

        Fully tensor-vectorised: cosine of every slot against the query,
        top-k selection, occupancy filtering, TTL boost/decay and eviction
        are batched GPU ops (the notebook's Python loop cost ~50 ms per
        decode step on a 512-slot bank; this version is sub-millisecond).

        Args:
            query: ``(1, L, d_model)`` batch of query vectors.
            count: maximal number of slots to retrieve.

        Returns:
            ``(relevant, n)`` where ``relevant`` is ``(1, n, d_model)``.
            Retrieved slots get +2 TTL (capped at 6); every TTL then decays
            by 1 and expired slots are evicted.
        """
        if query.shape[-1] != self.d_model:
            raise ValueError(f"query dim {query.shape[-1]} != d_model {self.d_model}")
        count = min(count, self.size)
        self._to(query.device, torch.float32)
        self.stats["get_calls"] += 1

        ctx_n = _unit(self.context, dim=2)
        q_n = _unit(query.float(), dim=2)
        best = torch.matmul(ctx_n, q_n.transpose(-2, -1)).max(dim=2)[0][0]   # (size,)
        _, idxs = torch.topk(best, k=count, largest=True)                     # (count,)

        norms = torch.linalg.vector_norm(self.context[0], ord=2, dim=1)
        valid = self.occupied[idxs] & (norms[idxs] > _NORM_EPS)
        chosen = idxs[valid]                                                   # ascending by score
        relevant = self.context[0, chosen].unsqueeze(0)                        # (n, 1, d) -> (1, n, d)

        # TTL: boost retrieved (+2 capped at 6), decay all by 1, evict <= 0.
        boosted = torch.clamp(self.ttl[chosen] + self.TTL_HIT_BONUS, max=self.TTL_MAX)
        self.ttl.scatter_(0, chosen, boosted)
        self.ttl -= 1
        expired = (self.ttl <= 0) & self.occupied
        n_ev = int(expired.sum())
        if n_ev:
            self.context[0, expired] = 0
            self.ttl[expired] = 0
            self.occupied[expired] = False
            self.stats["evictions"] += n_ev
        self.count_element = int(self.occupied.sum())

        self.stats["retrievals"] += chosen.numel()
        return relevant, int(chosen.numel())

    # ------------------------------------------------------------- ingestion
    @staticmethod
    def consume(master: torch.Tensor, slave: torch.Tensor) -> torch.Tensor:
        """Combine two vectors stored in the same slot (notebook's mean)."""
        return (master + slave) / 2

    def _merge_slot(self, master: torch.Tensor, slave: torch.Tensor) -> torch.Tensor:
        """Slot merge: plain mean (v1) or EMA ``a <- (1-k)a + k b`` (v2).

        The EMA form keeps the slot norm bounded while letting the new
        state nudge the old one instead of destroying it.
        """
        if self.merge_k is None:
            return self.consume(master, slave)
        return (1.0 - self.merge_k) * master + self.merge_k * slave

    @staticmethod
    def check(vector: torch.Tensor, window_size: int = 32, crit: float = 10.0) -> bool:
        """Activity gate: True when any window of ``vector`` sums above ``crit``."""
        dim_size = vector.shape[0]
        for i in range(0, dim_size, window_size):
            if vector[i : i + window_size].sum() > crit:
                return True
        return False

    def update(
        self,
        new: torch.Tensor,
        crit: float = 0.8,
        max_to_new: float = 0.4,
        window_size: int = 32,
        window_crit: float = 10.0,
    ) -> int:
        """Merge or store new vectors; returns the number of stored vectors.

        Vectorised but semantics-exact: several vectors landing in one slot
        in a single call are combined with the same geometric weights the
        sequential rule produces (``a <- (1-k)a + k b`` applied in order),
        computed with per-group coefficients instead of a Python loop.

        Args:
            new: ``(1, L, d_model)`` (or ``(L, d_model)``) vectors to ingest.
            crit: merge threshold — similarity at/above it merges into the
                best matching slot instead of allocating a new one.
            max_to_new: storage threshold — only vectors whose best match is
                at/below this similarity may claim a free slot.
        """
        if new.dim() == 3:
            new = new.squeeze(0)
        if new.shape[-1] != self.d_model:
            raise ValueError(f"new dim {new.shape[-1]} != d_model {self.d_model}")
        self._to(new.device, torch.float32)
        new = new.float()

        ctx_n = _unit(self.context[0], dim=1)
        q_n = _unit(new, dim=1)
        sims = ctx_n @ q_n.T                                       # (size, L)
        best_sim, best_idx = torch.max(sims, dim=0)

        self.stats["update_calls"] += 1

        occupied_best = self.occupied[best_idx]
        merge_ok = (best_sim >= crit) & occupied_best
        store_cand = (~merge_ok) & (best_sim <= max_to_new)
        if bool(torch.any(store_cand)):
            idxs = torch.nonzero(store_cand, as_tuple=True)[0]
            gated = self._check_batch(new[idxs], window_size, window_crit)
            keep = idxs[gated]
            self.stats["gate_rejects"] += int(idxs.numel() - keep.numel())
        else:
            keep = torch.zeros(0, dtype=torch.long, device=self.device)
            self.stats["gate_rejects"] += 0
        store_ok = torch.zeros_like(store_cand)
        store_ok[keep] = True

        m_idx = torch.nonzero(merge_ok, as_tuple=True)[0]
        if m_idx.numel():
            self._apply_merges(best_idx[m_idx], new[m_idx])
        s_idx = torch.nonzero(store_ok, as_tuple=True)[0]
        if s_idx.numel():
            free = torch.nonzero(~self.occupied, as_tuple=True)[0]
            s_idx = s_idx[: free.numel()]          # original stops storing when full
            if s_idx.numel():
                self.context[0, free[: s_idx.numel()]] = new[s_idx]
                self.ttl[free[: s_idx.numel()]] = self.TTL_NEW
                self.occupied[free[: s_idx.numel()]] = True
                self.stats["stores"] += int(s_idx.numel())
        self.count_element = int(self.occupied.sum())
        return int(s_idx.numel())

    def _apply_merges(self, slots: torch.Tensor, vecs: torch.Tensor) -> None:
        """Sequential merge semantics in one grouped pass."""
        order = torch.argsort(slots, stable=True)
        slots_s, vecs_s = slots[order], vecs[order]
        counts = torch.bincount(slots_s, minlength=self.size)
        group_ids = slots_s
        # position of each item within its slot group (0-based, arrival order)
        ends = counts.cumsum(0)[group_ids]
        starts = ends - counts[group_ids]
        ar = torch.arange(slots_s.numel(), device=self.device)
        pos = ar - starts                                  # i-th merge in group
        m = counts[group_ids]                              # group size
        k = self.merge_k if self.merge_k is not None else 0.5
        b_coef = k * (1.0 - k) ** (m - pos - 1)            # weight of vecs_s[j]
        acc = torch.zeros_like(self.context[0])
        acc.index_add_(0, slots_s, vecs_s * b_coef[:, None])
        uniq = torch.nonzero(counts, as_tuple=True)[0]
        a_coef = (1.0 - k) ** counts[uniq]
        self.context[0, uniq] = self.context[0, uniq] * a_coef[:, None] + acc[uniq]
        self.ttl[uniq] = torch.maximum(self.ttl[uniq], torch.full_like(self.ttl[uniq], float(self.TTL_NEW)))
        self.stats["merges"] += int(slots_s.numel())

    @staticmethod
    def _check_batch(vecs: torch.Tensor, window_size: int, crit: float) -> torch.Tensor:
        """Vectorised activity gate over a batch of rows (loop-semantics exact)."""
        d = vecs.shape[-1]
        n = d // window_size
        if n == 0:
            return vecs.sum(dim=1) > crit  # whole row is one (short) window
        full = vecs[:, : n * window_size].reshape(-1, n, window_size).sum(dim=2)
        ok = full.max(dim=1).values > crit
        if n * window_size < d:  # ragged tail counts as a window, like the loop version
            ok = ok | (vecs[:, n * window_size :].sum(dim=1) > crit)
        return ok

    def _slot_norms(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.context[0], ord=2, dim=1)


class CorrectionLayer(nn.Module):
    """nn.Module wrapper: retrieve relevant context, then ingest ``x``.

    Used as a forward-pre-hook payload on a transformer block: given the
    hidden states it returns the memory context relevant to them.
    """

    def __init__(
        self,
        d_model: int,
        size: int = 512,
        retrieve_count: int = 256,
        window_crit: float = 10.0,
        crit: float = 0.8,
        max_to_new: float = 0.4,
        merge_k: float | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.retrieve_count = retrieve_count
        self.window_crit = window_crit
        self.crit = crit
        self.max_to_new = max_to_new
        self.memory = Memory(d_model, size=size, merge_k=merge_k)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Retrieve then update; returns ``(relevant, actual_count)``."""
        relevant, actual_count = self.memory.get_relevant(x, self.retrieve_count)
        self.memory.update(x, crit=self.crit, max_to_new=self.max_to_new, window_crit=self.window_crit)
        return relevant, actual_count

    @property
    def stats(self) -> dict:
        """Behavioural counters of the underlying memory (hits, merges, ...)."""
        return dict(self.memory.stats, occupied_slots=int(self.memory.count_element))

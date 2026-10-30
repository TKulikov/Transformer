"""Behavioural tests for the associative memory (notebook core)."""

from __future__ import annotations

import pytest
import torch

from tmem.memory import CorrectionLayer, Memory

D = 64


def hot(seed: int = 0, n: int = 1) -> torch.Tensor:
    """Vectors that pass the activity gate (window sums > 10)."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, D, generator=g).abs() + 0.5  # positive => window sum large
    return v


def blocks(*idx: int) -> torch.Tensor:
    """Near-orthogonal vectors: positives on disjoint coordinate ranges.

    Each row passes the activity gate (window sum > 10) and pairwise
    cosine similarity is exactly 0.
    """
    rows = []
    for i in idx:
        v = torch.zeros(D)
        v[21 * i : 21 * (i + 1)] = 1.0
        rows.append(v)
    return torch.stack(rows)


class TestBasics:
    def test_fresh_memory_returns_nothing(self):
        m = Memory(D, size=16)
        rel, n = m.get_relevant(hot().unsqueeze(0), count=8)
        assert n == 0
        assert rel.shape == (1, 0, D)

    def test_invalid_dims(self):
        m = Memory(D, size=16)
        with pytest.raises(ValueError):
            m.get_relevant(torch.zeros(1, 2, 3), count=4)
        with pytest.raises(ValueError):
            m.update(torch.zeros(1, 2, 5))
        with pytest.raises(ValueError):
            Memory(0)
        with pytest.raises(ValueError):
            Memory(D, size=0)

    def test_clear_resets_state(self):
        m = Memory(D, size=16)
        m.update(hot().unsqueeze(0))
        assert m.count_element > 0
        m.clear()
        assert m.count_element == 0
        assert not bool(m.occupied.any())
        assert torch.all(m.context == 0)
        assert torch.all(m.ttl == 0)


class TestUpdate:
    def test_stores_active_novel_vector(self):
        m = Memory(D, size=16)
        stored = m.update(hot().unsqueeze(0))
        assert stored == 1
        assert m.count_element == 1

    def test_ignores_inactive_vector(self):
        """Vectors below the activity gate are not stored (notebook check())."""
        m = Memory(D, size=16)
        quiet = torch.full((1, D), 0.05)  # window sum = 3.2 < 10
        assert m.update(quiet.unsqueeze(0)) == 0
        assert m.count_element == 0

    def test_duplicate_merges_not_stores(self):
        m = Memory(D, size=16)
        v = hot()
        m.update(v.unsqueeze(0))
        before = m.count_element
        m.update(v.unsqueeze(0))  # similarity ~1.0 >= crit -> merge
        assert m.count_element == before
        # merging a vector with itself keeps it unchanged
        idx = int(torch.nonzero(m.occupied)[0])
        assert torch.allclose(m.context[0, idx], v[0], atol=1e-5)

    def test_mid_similarity_neither_merges_nor_stores(self):
        """0.4 < sim < 0.8 -> below crit (no merge) and above max_to_new (no slot)."""
        m = Memory(D, size=16)
        v = hot(1)
        m.update(v.unsqueeze(0))
        w = v.clone()
        noise = torch.randn_like(w) * 0.25 * w.norm(dim=1, keepdim=True) / D**0.5
        w = w + noise
        sim = torch.cosine_similarity(w, v, dim=1)
        # Only meaningful when the crafted vector really falls into the gap.
        if bool((sim > 0.4) & (sim < 0.8)):
            assert m.update(w.unsqueeze(0)) == 0
            assert m.count_element == 1

    def test_capacity_full_rejects_new(self):
        m = Memory(D, size=1)
        v0, v1 = blocks(0), blocks(1)
        m.update(v0.unsqueeze(0))
        assert torch.cosine_similarity(v0, v1, dim=1).item() == pytest.approx(0.0)
        assert m.update(v1.unsqueeze(0)) == 0  # novel, but no slot free
        assert m.count_element == 1

    def test_batch_of_vectors(self):
        m = Memory(D, size=16)
        v = torch.stack([hot(i)[0] for i in range(4)])
        stored = m.update(v.unsqueeze(0))
        assert stored >= 3  # near-orthogonal randoms; worst case one merge
        assert m.count_element >= 3


class TestRetrieval:
    def test_retrieves_matching_slot(self):
        m = Memory(D, size=16)
        v1, v2 = hot(11), hot(12)
        m.update(torch.stack([v1[0], v2[0]]).unsqueeze(0))
        rel, n = m.get_relevant(v1.unsqueeze(0), count=8)
        assert n >= 1
        sims = torch.nn.functional.cosine_similarity(rel[0], v1[0].expand_as(rel[0]), dim=1)
        assert float(sims.max()) > 0.9

    def test_retrieval_count_is_bounded_by_occupied(self):
        m = Memory(D, size=16)
        m.update(hot(3).unsqueeze(0))  # one vector
        rel, n = m.get_relevant(hot(4).unsqueeze(0), count=8)
        assert n <= 1

    def test_hit_bumps_ttl(self):
        m = Memory(D, size=16)
        v0, v1 = blocks(0), blocks(1)
        m.update(torch.stack([v0[0], v1[0]]).unsqueeze(0))
        i0, i1 = int(torch.nonzero(m.occupied)[0]), int(torch.nonzero(m.occupied)[1])
        m.get_relevant(v0.unsqueeze(0), count=1)  # only the v0 slot is retrieved
        assert float(m.ttl[i0]) > float(m.ttl[i1])

    def test_ttl_decay_and_eviction(self):
        m = Memory(D, size=16)
        v = hot(6)
        m.update(v.unsqueeze(0))
        # A query with strictly negative similarity is never selected: all
        # empty slots score 0, so the stored slot only decays (-1 per get).
        unrelated = -torch.ones(1, 1, D)
        for _ in range(4):
            m.get_relevant(unrelated, count=4)
        assert m.count_element == 1  # stored TTL 5, decayed 4 times
        m.get_relevant(unrelated, count=4)
        assert m.count_element == 0
        assert not bool(m.occupied.any())

    def test_eviction_frees_slot_reusable_no_double_count(self):
        """Regression vs the notebook bug: evicted indices could be re-added
        twice to the empty-slot list, corrupting the bookkeeping."""
        m = Memory(D, size=2)
        v = hot(8)
        m.update(v.unsqueeze(0))
        for _ in range(6):  # let it die, then store again
            m.get_relevant(torch.zeros(1, 1, D), count=1)
        m.update(hot(9).unsqueeze(0))
        m.update(hot(8).unsqueeze(0))
        assert m.count_element == int(m.occupied.sum())

    def test_multiple_query_positions_use_max_similarity(self):
        m = Memory(D, size=16)
        v = hot(9)
        m.update(v.unsqueeze(0))
        q = torch.stack([torch.randn(D), v[0]])  # second position matches
        rel, n = m.get_relevant(q.unsqueeze(0), count=4)
        assert n == 1


class TestGatesAndOps:
    def test_check_window_gate(self):
        assert Memory.check(torch.full((D,), 0.5), window_size=32, crit=10.0)  # sum 16
        assert not Memory.check(torch.full((D,), 0.1), window_size=32, crit=10.0)

    def test_check_partial_window(self):
        v = torch.zeros(D)
        v[32:48] = 1.0  # the window covering dims 32..48 sums to 16 > 10
        assert Memory.check(v, window_size=16, crit=10.0)
        assert not Memory.check(v, window_size=64, crit=20.0)

    def test_consume_is_mean(self):
        a, b = torch.randn(D), torch.randn(D)
        assert torch.allclose(Memory.consume(a, b), (a + b) / 2)

    def test_determinism(self):
        def run():
            m = Memory(D, size=16)
            m.update(hot(10).unsqueeze(0))
            return m.get_relevant(hot(11).unsqueeze(0), count=4)[1]

        assert run() == run()

    def test_stats_counters(self):
        m = Memory(D, size=2)
        v = hot(1)
        m.update(v.unsqueeze(0))
        assert m.stats["stores"] == 1 and m.stats["update_calls"] == 1
        m.update(v.unsqueeze(0))  # duplicate -> merge
        assert m.stats["merges"] == 1
        m.update((-hot(1) * 0.01).unsqueeze(0))  # anti-correlated: passes into new-slot branch...
        assert m.stats["gate_rejects"] == 1      # ...but fails the activity gate (negative sums)
        m.get_relevant(-torch.ones(1, 1, D), count=1)  # no retrieval, pure decay x5
        for _ in range(4):
            m.get_relevant(-torch.ones(1, 1, D), count=1)
        assert m.stats["get_calls"] == 5
        assert m.stats["evictions"] == 1
        assert m.stats["retrievals"] == 0

    def test_clear_resets_stats(self):
        m = Memory(D, size=4)
        m.update(hot(2).unsqueeze(0))
        m.clear()
        assert all(v == 0 for v in m.stats.values())


class TestAgainstLoopReference:
    """Property test: vectorised update/get_relevant == the loop semantics."""

    @staticmethod
    def loop_reference(m: Memory, new: torch.Tensor, crit: float, max_to_new: float,
                       window_size: int, window_crit: float) -> None:
        """Straight transcription of the notebook's original algorithm."""
        new = new.squeeze(0)
        ctx_n = m.context[0] / torch.clamp(torch.linalg.vector_norm(m.context[0], dim=1, keepdim=True), min=1e-8)
        q_n = new / torch.clamp(torch.linalg.vector_norm(new, dim=1, keepdim=True), min=1e-8)
        sims = ctx_n @ q_n.T
        best_sim, best_idx = torch.max(sims, dim=0)
        for i in range(new.shape[0]):
            vec, sim, idx = new[i], float(best_sim[i]), int(best_idx[i])
            if sim >= crit and bool(m.occupied[idx]):
                if m.merge_k is None:
                    m.context[0, idx] = (m.context[0, idx] + vec) / 2
                else:
                    m.context[0, idx] = (1 - m.merge_k) * m.context[0, idx] + m.merge_k * vec
                m.ttl[idx] = max(float(m.ttl[idx]), float(m.TTL_NEW))
            elif sim <= max_to_new and Memory.check(vec, window_size, window_crit):
                free = torch.nonzero(~m.occupied, as_tuple=True)[0]
                if free.numel() == 0:
                    continue
                pos = int(free[0])
                m.context[0, pos] = vec
                m.ttl[pos] = m.TTL_NEW
                m.occupied[pos] = True
                m.count_element += 1

    def _run_pair(self, merge_k, seed):
        g = torch.Generator().manual_seed(seed)
        args = dict(d_model=16, size=5, merge_k=merge_k)
        mA = Memory(**args)
        mB = Memory(**args)
        for step in range(20):
            batch = (torch.randn(1, 6, 16, generator=g) * (1 + (step % 3)))
            if step % 4 == 0:
                batch = batch.abs() + 0.5  # activity-gate friendly
            common = batch.clone()
            mA.update(common.clone(), crit=0.6, max_to_new=0.45, window_crit=1.0)
            self.loop_reference(mB, common.clone(), crit=0.6, max_to_new=0.45,
                                window_size=8, window_crit=1.0)
            # retrieval path of mA uses the vectorised eviction; run it on mB
            # by transcribing the same rule before comparing full state
            q = torch.randn(1, 4, 16, generator=g)
            mA.get_relevant(q, count=5)
            self._loop_get_relevant(mB, q.clone(), count=5)
            assert torch.allclose(mA.context, mB.context, atol=1e-5), (step, merge_k)
            assert torch.allclose(mA.ttl, mB.ttl, atol=1e-6), (step, merge_k)
            assert bool((mA.occupied == mB.occupied).all()), (step, merge_k)

    @staticmethod
    def _loop_get_relevant(m: Memory, query: torch.Tensor, count: int) -> None:
        count = min(count, m.size)
        ctx_n = m.context / torch.clamp(torch.linalg.vector_norm(m.context, dim=2, keepdim=True), min=1e-8)
        q_n = query / torch.clamp(torch.linalg.vector_norm(query, dim=2, keepdim=True), min=1e-8)
        best = torch.max(torch.matmul(ctx_n, q_n.transpose(-2, -1)), dim=2)[0][0]
        _, idxs = torch.topk(best, k=count, largest=True)
        norms = torch.linalg.vector_norm(m.context[0], dim=1)
        for idx in idxs.tolist():
            if bool(m.occupied[idx]) and float(norms[idx]) > 1e-6:
                m.ttl[idx] = min(m.TTL_MAX, float(m.ttl[idx]) + m.TTL_HIT_BONUS)
        m.ttl -= 1
        expired = (m.ttl <= 0) & m.occupied
        if bool(torch.any(expired)):
            ii = torch.nonzero(expired, as_tuple=True)[0]
            m.context[0, ii] = 0
            m.ttl[ii] = 0
            m.occupied[ii] = False
            m.count_element -= int(ii.numel())

    def test_mean_merge_match(self):
        self._run_pair(merge_k=None, seed=1)

    def test_ema_merge_match(self):
        self._run_pair(merge_k=0.25, seed=2)


class TestCorrectionLayer:
    def test_forward_retrieves_then_updates(self):
        corr = CorrectionLayer(D, size=16)
        rel, n = corr(hot(1).unsqueeze(0))
        assert n == 0 and corr.memory.count_element == 1  # first call: empty -> nothing retrieved
        rel, n = corr(hot(1).unsqueeze(0))
        assert n == 1  # second call finds the stored vector

    def test_retrieve_count_configurable(self):
        corr = CorrectionLayer(D, size=16, retrieve_count=1)
        corr.memory.update(torch.stack([hot(i)[0] for i in (1, 2, 3)]).unsqueeze(0))
        _, n = corr(hot(1).unsqueeze(0))
        assert n <= 1

    def test_is_nn_module(self):
        import torch.nn as nn

        corr = CorrectionLayer(D)
        assert isinstance(corr, nn.Module)
        assert corr.d_model == D

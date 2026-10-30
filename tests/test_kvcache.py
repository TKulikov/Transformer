"""Tests for cache policies, the compat shim and importance tracking."""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from tmem.compat import cache_length, layer_kv, prune_cache, self_cache
from tmem.kvcache import ImportanceTracker, KVConfig, keep_indices


class TestKVConfig:
    def test_invalid_mode_strategy(self):
        with pytest.raises(ValueError):
            KVConfig(mode="magic")
        with pytest.raises(ValueError):
            KVConfig(mode="sparse", strategy="magic")

    def test_budget_must_cover_sink(self):
        with pytest.raises(ValueError):
            KVConfig(mode="sparse", budget=3, sink=4)

    def test_factories(self):
        assert KVConfig.classic().mode == "classic"
        sp = KVConfig.sparse(64, "topk", sink=2)
        assert sp.budget == 64 and sp.strategy == "topk" and sp.needs_scores
        assert not KVConfig.classic().needs_scores
        assert not KVConfig.sparse(64, "sliding").needs_scores


class TestKeepIndices:
    def test_classic_or_under_budget_keeps_all(self):
        cfg = KVConfig.classic()
        assert torch.equal(keep_indices(10, cfg), torch.arange(10))
        sparse = KVConfig.sparse(16)
        assert torch.equal(keep_indices(16, sparse), torch.arange(16))

    def test_sliding_keeps_sink_and_recent_tail(self):
        cfg = KVConfig.sparse(10, "sliding", sink=2, recent=4)
        keep = keep_indices(25, cfg)
        assert keep.numel() == 10
        assert keep[0] == 0 and keep[1] == 1          # sink
        assert keep[-1] == 24                          # newest token always kept
        assert torch.equal(keep, keep.sort().values)   # ascending
        assert (keep < 2).sum() == 2 and (keep >= 25 - 8).sum() == 8

    def test_topk_selects_high_importance_middle(self):
        cfg = KVConfig.sparse(12, "topk", sink=2, recent=4)
        importance = torch.arange(30, dtype=torch.float32)  # newest middle most important
        keep = keep_indices(30, cfg, importance)
        assert keep.numel() == 12
        middle = keep[(keep >= 2) & (keep < 26)]
        assert middle.numel() == 6
        assert int(middle.min()) >= 20  # only the strongest middle slots survived

    def test_topk_requires_scores(self):
        cfg = KVConfig.sparse(8, "topk", sink=1, recent=2)
        with pytest.raises(ValueError):
            keep_indices(20, cfg, importance=None)

    def test_topk_degenerates_when_recent_fills_budget(self):
        cfg = KVConfig.sparse(6, "topk", sink=2, recent=4)
        keep = keep_indices(20, cfg, torch.rand(20))
        assert keep.numel() == 6


class TestImportanceTracker:
    def test_accumulates_column_mass(self):
        t = ImportanceTracker()
        a1 = torch.ones(1, 2, 3, 3)  # (bsz, heads, q, k): every query spreads 1
        t.step((a1,))
        assert torch.allclose(t.scores, torch.tensor([6.0, 6.0, 6.0]))  # 2 heads * 3 queries
        t.step((a1, a1))  # two layers, mean over layers
        assert torch.allclose(t.scores, torch.tensor([12.0, 12.0, 12.0]))

    def test_grows_with_cache(self):
        t = ImportanceTracker()
        t.step((torch.ones(1, 1, 2, 2),))
        t.step((torch.ones(1, 1, 1, 3),))  # one new slot
        assert t.scores.numel() == 3
        assert t.scores[2].item() == pytest.approx(1.0)

    def test_prune_renumbers(self):
        t = ImportanceTracker()
        t.scores = torch.arange(10, dtype=torch.float32)
        t.prune(torch.tensor([0, 5, 9]))
        assert torch.equal(t.scores, torch.tensor([0.0, 5.0, 9.0]))

    def test_none_attentions_is_noop(self):
        t = ImportanceTracker()
        t.step(None)
        t.step((None,))
        assert t.scores.numel() == 0


def make_cache(n_layers: int = 2, seq: int = 5):
    cache = DynamicCache()
    for layer in range(n_layers):
        keys = torch.arange(seq, dtype=torch.float32).view(1, 1, seq, 1) + layer * 100
        cache.update(keys, keys * 10, layer)
    return cache


class TestCompatShim:
    def test_length_and_layers(self):
        cache = make_cache()
        assert cache_length(cache) == 5
        assert len(layer_kv(cache)) == 2
        keys, values = layer_kv(cache)[1]
        assert keys.shape == (1, 1, 5, 1)
        assert torch.allclose(values, keys * 10)

    def test_prune_keeps_selected_slots(self):
        cache = make_cache(seq=6)
        keep = torch.tensor([0, 2, 5])
        prune_cache(cache, keep)
        assert cache_length(cache) == 3
        keys, _ = layer_kv(cache)[0]
        assert torch.equal(keys.view(-1), torch.tensor([0.0, 2.0, 5.0]))

    def test_prune_out_of_range_raises(self):
        cache = make_cache(seq=3)
        with pytest.raises(IndexError):
            prune_cache(cache, torch.tensor([0, 4]))

    def test_self_cache_passthrough_and_encdec(self):
        cache = DynamicCache()
        assert self_cache(cache) is cache
        encdec = DynamicCache()  # plain caches must also work through self_cache
        assert cache_length(self_cache(encdec)) == 0

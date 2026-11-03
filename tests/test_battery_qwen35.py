"""Tests for the >100-turn battery and Qwen3.5 integration plumbing."""

from __future__ import annotations

import torch

from tmem.compat import cache_length
from tmem.models import MODEL_REGISTRY, tiny_causal
from tmem.stress import battery_strategies, build_battery_100


class TestBattery100:
    def test_structure(self):
        session = build_battery_100()
        turns = session.user_turns()
        assert len(turns) >= 100
        assert len(session.probes) == 24
        assert sum(p.paraphrase for p in session.probes) == 12
        for p in session.probes:
            assert p.plant_turn < p.after_turn
            assert p.expected
        plants = [p.plant_turn for p in session.probes]
        total = len(turns)
        assert min(plants) < total / 3 and max(plants) > total / 3  # facts across zones

    def test_updates_present(self):
        session = build_battery_100()
        assert any(p.stale for p in session.probes)

    def test_battery_grid_runs(self, causal_bundle):
        from tmem.dialogue import run_session

        session = build_battery_100(target_turns=105)
        grid = battery_strategies(budget=128)
        # KV injection wraps Qwen3.5 attention specifically, and PyramidKV
        # needs per-layer masks that GPT-2 does not provide.
        names = {
            "classic_kv_full", "sparse_sliding_128", "sparse_topk_128",
            "snapkv_128", "rag_text_only",
        }
        pick = [s for s in grid if s.name in names]
        assert len(pick) == 5
        for strategy in pick:
            run = run_session(causal_bundle, session, strategy, max_new_tokens=2, seed=0)
            s = run.summary()
            assert s["turns"] >= 100 and s["probe_mean"] is not None


class TestQwen35Plumbing:
    def test_registry_entry(self):
        assert "qwen35-4b" in MODEL_REGISTRY
        assert MODEL_REGISTRY["qwen35-4b"]["repo"] == "Qwen/Qwen3.5-4B"

    def test_to_chat_no_template_fallback(self, causal_bundle):
        assert causal_bundle.to_chat("plain text") == "plain text"

    def test_eos_ids_flattens_lists(self):
        bundle = tiny_causal()
        bundle.model.config.eos_token_id = [1, 2, 3]
        assert set(bundle.eos_ids()) >= {1, 2, 3}

    def test_cache_length_skips_non_kv_layers(self):
        class FakeLayer:
            def __init__(self, keys=None):
                self.keys, self.values = keys, keys

        class FakeCache:
            def __init__(self, layers):
                self.layers = layers

        cache = FakeCache([FakeLayer(None), FakeLayer(None), FakeLayer(torch.zeros(1, 2, 7, 3))])
        assert cache_length(cache) == 7
        assert cache_length(FakeCache([FakeLayer(None)])) == 0

    def test_memory_target_finds_blocks(self):
        bundle = tiny_causal()
        bundle.name = "unknown-model"  # forces the dynamic resolution path
        block = bundle.memory_target()
        assert block is not None

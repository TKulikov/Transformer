"""Tests for the decoding loop: correctness of caches, budgeting, sampling."""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from tmem.generate import decode_causal, decode_encdec, pick_token
from tmem.kvcache import KVConfig


def ids(text_len: int = 12, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, 256, (1, text_len), generator=g)


class TestCausalCacheEquivalence:
    """The KV cache must not change the model's distribution — classic cache
    decoding must reproduce cache-less recompute token-for-token (the KV
    cache correctness baseline)."""

    def test_classic_matches_no_cache(self, causal_bundle):
        model = causal_bundle.model
        a = decode_causal(model, ids(), cache="none", max_new_tokens=8, seed=0)
        b = decode_causal(model, ids(), kv=KVConfig.classic(), cache="use", max_new_tokens=8, seed=0)
        assert a.token_ids == b.token_ids
        assert b.cache_lengths[0] == a.prompt_len

    @pytest.mark.parametrize("strategy", ["sliding", "topk", "snapkv", "pyramidkv"])
    def test_sparse_with_huge_budget_matches_classic(self, causal_bundle, strategy):
        model = causal_bundle.model
        classic = decode_causal(model, ids(), kv=KVConfig.classic(), cache="use", max_new_tokens=8, seed=0)
        roomy = decode_causal(model, ids(), kv=KVConfig.sparse(1024, strategy), cache="use", max_new_tokens=8, seed=0)
        assert classic.token_ids == roomy.token_ids

    def test_classic_cache_lengths_grow(self, causal_bundle):
        res = decode_causal(causal_bundle.model, ids(10), kv=KVConfig.classic(), cache="use", max_new_tokens=6, seed=0)
        assert res.cache_lengths == [10, 11, 12, 13, 14, 15][: len(res.cache_lengths)] or res.cache_lengths == list(range(10, 10 + res.steps))


class TestSparseBudgets:
    @pytest.mark.parametrize("strategy", ["sliding", "topk"])
    def test_cache_stays_within_budget(self, causal_bundle, strategy):
        kv = KVConfig.sparse(budget=16, strategy=strategy, sink=2, recent=8)
        res = decode_causal(causal_bundle.model, ids(20), kv=kv, cache="use", max_new_tokens=10, seed=0)
        assert res.steps == 10
        assert max(res.cache_lengths) <= 16
        assert res.cache_lengths[-1] == 16
        assert res.total_cache_slots == sum(res.cache_lengths)

    @pytest.mark.parametrize("strategy", ["sliding", "topk"])
    def test_sparse_produces_full_generation(self, causal_bundle, strategy):
        kv = KVConfig.sparse(budget=12, strategy=strategy, sink=1, recent=4)
        res = decode_causal(causal_bundle.model, ids(14), kv=kv, cache="use", max_new_tokens=12, seed=0)
        assert res.steps == 12  # pruning must never stall decoding

    def test_sparse_saves_attention_work_vs_classic(self, causal_bundle):
        long_prompt = ids(60, seed=5)
        classic = decode_causal(causal_bundle.model, long_prompt, kv=KVConfig.classic(), cache="use", max_new_tokens=20, seed=0)
        sparse = decode_causal(causal_bundle.model, long_prompt, kv=KVConfig.sparse(32, "sliding", sink=4, recent=16), cache="use", max_new_tokens=20, seed=0)
        assert sparse.total_cache_slots < classic.total_cache_slots

    def test_snapkv_compresses_prompt_before_generation(self, causal_bundle):
        kv = KVConfig.sparse(budget=16, strategy="snapkv", observation=4)
        res = decode_causal(causal_bundle.model, ids(40), kv=kv, cache="use", max_new_tokens=6, seed=0)
        assert res.steps == 6
        assert res.cache_lengths[0] == 16
        assert max(res.cache_lengths) <= 16 + res.steps - 1
        assert max(res.cache_max_lengths) <= 16 + res.steps - 1

    def test_pyramidkv_rejects_shared_mask_backend_after_compression(self, causal_bundle):
        kv = KVConfig.sparse(budget=16, strategy="pyramidkv", observation=4)
        with pytest.raises(RuntimeError, match="requires per-layer attention masks"):
            decode_causal(causal_bundle.model, ids(40), kv=kv, cache="use", max_new_tokens=6, seed=0)


class TestStoppingAndSampling:
    def test_eos_stops_early(self, causal_bundle):
        reference = decode_causal(causal_bundle.model, ids(), cache="use", max_new_tokens=8, seed=0)
        forced_eos = reference.token_ids[2]
        first_at = reference.token_ids.index(forced_eos) + 1  # handles repeated tokens
        stopped = decode_causal(causal_bundle.model, ids(), cache="use", max_new_tokens=8, seed=0, eos_ids=(forced_eos,))
        assert stopped.token_ids == reference.token_ids[:first_at]

    def test_sampling_is_seeded(self, causal_bundle):
        a = decode_causal(causal_bundle.model, ids(), cache="use", max_new_tokens=10, do_sample=True, seed=42)
        b = decode_causal(causal_bundle.model, ids(), cache="use", max_new_tokens=10, do_sample=True, seed=42)
        assert a.token_ids == b.token_ids

    def test_greedy_vs_sampling_diverge_or_not(self, causal_bundle):
        g = decode_causal(causal_bundle.model, ids(), cache="use", max_new_tokens=6, do_sample=False)
        s = decode_causal(causal_bundle.model, ids(), cache="use", max_new_tokens=6, do_sample=True, seed=1)
        assert len(g.token_ids) == len(s.token_ids) == 6

    def test_pick_token_greedy_argmax(self):
        logits = torch.zeros(1, 10)
        logits[0, 7] = 5.0
        assert pick_token(logits, do_sample=False, temperature=0.7, top_p=1.0, generator=None) == 7

    def test_pick_token_sample_temperature_error(self):
        with pytest.raises(ValueError):
            pick_token(torch.zeros(1, 4), do_sample=True, temperature=0.0, top_p=1.0, generator=None)


class TestEncDec:
    def test_classic_matches_no_cache(self, encdec_bundle):
        enc_ids = ids(9, seed=4)
        a = decode_encdec(encdec_bundle.model, enc_ids, decoder_start_id=0, cache="none", max_new_tokens=6, seed=0)
        b = decode_encdec(encdec_bundle.model, enc_ids, decoder_start_id=0, cache="use", max_new_tokens=6, seed=0)
        assert a.token_ids == b.token_ids

    @pytest.mark.parametrize("strategy", ["sliding", "topk"])
    def test_sparse_bounds_self_cache_only(self, encdec_bundle, strategy):
        kv = KVConfig.sparse(budget=10, strategy=strategy, sink=1, recent=4)
        res = decode_encdec(encdec_bundle.model, ids(8, seed=2), decoder_start_id=0, kv=kv, cache="use", max_new_tokens=10, seed=0)
        assert max(res.cache_lengths) <= 10
        assert res.steps == 10

    def test_memory_hook_integration(self, encdec_bundle):
        """Encoder-injection changes hidden states mid-forward; the loop must
        keep the cross-attention mask consistent with the extended encoder."""
        corr_hook_bundle = encdec_bundle
        hook, corr = corr_hook_bundle.make_memory_hook(size=8)
        corr.memory.update(torch.ones(1, 4, corr_hook_bundle.hidden_size))  # pre-fill so retrieval is non-empty
        handle = corr_hook_bundle.memory_target().register_forward_pre_hook(hook, with_kwargs=True)
        try:
            res = decode_encdec(corr_hook_bundle.model, ids(7, seed=9), decoder_start_id=0, cache="use", max_new_tokens=4, seed=0)
            assert res.steps == 4
        finally:
            handle.remove()

    def test_rejects_bad_input_shape(self, causal_bundle):
        with pytest.raises(ValueError):
            decode_causal(causal_bundle.model, torch.randint(0, 5, (2, 4)), cache="use", max_new_tokens=2)

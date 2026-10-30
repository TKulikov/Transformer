"""Tests for the model registry, bundles and the offline byte tokenizer."""

from __future__ import annotations

import pytest
import torch

from tmem.memory import CorrectionLayer
from tmem.models import MODEL_REGISTRY, ByteTokenizer, load_model, resolve, tiny_causal, tiny_encdec


class TestRegistry:
    def test_known_models(self):
        assert set(MODEL_REGISTRY) == {"gpt2", "qwen2-1.5b", "qwen35-4b", "t5-base"}

    def test_resolve_kinds(self):
        assert resolve("GPT2")["kind"] == "causal"          # case-insensitive
        assert resolve("t5-base")["kind"] == "encdec"
        with pytest.raises(KeyError):
            resolve("llama")


class TestBundles:
    def test_hidden_sizes(self, causal_bundle, encdec_bundle):
        assert causal_bundle.hidden_size == 64
        assert encdec_bundle.hidden_size == 64

    def test_memory_targets(self, causal_bundle, encdec_bundle):
        assert causal_bundle.memory_target() is not None
        assert encdec_bundle.memory_target() is not None

    def test_hook_factory_matches_kind(self, causal_bundle, encdec_bundle):
        for bundle in (causal_bundle, encdec_bundle):
            hook, corr = bundle.make_memory_hook()
            assert isinstance(corr, CorrectionLayer)
            assert callable(hook)

    def test_tiny_models_are_deterministic(self):
        a, b = tiny_causal(seed=3), tiny_causal(seed=3)
        x = torch.randint(1, 100, (1, 6))
        out_a = a.model(input_ids=x).logits
        out_b = b.model(input_ids=x).logits
        assert torch.equal(out_a, out_b)

    def test_tiny_encdec_start_and_eos(self, encdec_bundle):
        assert encdec_bundle.decoder_start_id() == 0


class TestByteTokenizer:
    def test_roundtrip(self):
        tok = ByteTokenizer()
        ids = tok("hello world 42", truncation=True, max_length=512)["input_ids"]
        assert tok.decode(ids[0].tolist()) == "hello world 42"

    def test_truncation(self):
        tok = ByteTokenizer()
        ids = tok("abcdefghij", truncation=True, max_length=4)["input_ids"]
        assert ids.shape == (1, 4)


def test_load_model_unknown_raises():
    with pytest.raises(KeyError):
        load_model("nope")

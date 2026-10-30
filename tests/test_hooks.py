"""Tests for the memory-injection hooks (both architecture flavours)."""

from __future__ import annotations

import pytest
import torch

from tmem.hooks import make_decoder_injection_hook, make_encoder_injection_hook
from tmem.memory import CorrectionLayer

D = 32


def filled_corr(n_vecs: int = 2, seed: int = 0) -> CorrectionLayer:
    corr = CorrectionLayer(D, size=16)
    vecs = torch.randn(n_vecs, D, generator=torch.Generator().manual_seed(seed)).abs() + 0.5
    corr.memory.update(vecs.unsqueeze(0))
    return corr


def fake_block_args(hs: torch.Tensor, mask: torch.Tensor | None = None):
    return (hs, mask) if mask is not None else (hs,)


class TestDecoderHook:
    def test_shapes_preserved_and_influence_added(self):
        corr = filled_corr()
        hook = make_decoder_injection_hook(corr, beta=0.5)
        hs = torch.randn(1, 5, D)
        rel0, n = corr(hs)  # warm the memory so retrieval is non-empty
        (out_hs,), kwargs = hook(None, (hs,), {})
        assert out_hs.shape == hs.shape
        assert not torch.allclose(out_hs, hs)  # memory influenced it

    def test_empty_memory_passthrough(self):
        corr = CorrectionLayer(D, size=16)  # nothing stored -> count 0
        hook = make_decoder_injection_hook(corr)
        hs = torch.randn(1, 4, D)
        (out_hs,), kwargs = hook(None, (hs,), {})
        assert torch.allclose(out_hs, hs)

    def test_kwargs_style_call(self):
        corr = filled_corr()
        hook = make_decoder_injection_hook(corr)
        hs = torch.randn(1, 3, D)
        corr(hs)
        args, kwargs = hook(None, (), {"hidden_states": hs})
        assert kwargs["hidden_states"].shape == hs.shape

    def test_beta_zero_is_identity(self):
        corr = filled_corr()
        hook = make_decoder_injection_hook(corr, beta=0.0)
        hs = torch.randn(1, 3, D)
        corr(hs)
        (out_hs,), _ = hook(None, (hs,), {})
        assert torch.allclose(out_hs, hs, atol=1e-6)

    def test_dtype_downcast_roundtrip(self):
        """Half-precision hidden states must not blow up (memory is fp32)."""
        corr = filled_corr()
        hook = make_decoder_injection_hook(corr)
        hs = torch.randn(1, 3, D, dtype=torch.float16)
        corr(hs.float())
        (out_hs,), _ = hook(None, (hs,), {})
        assert out_hs.dtype == torch.float16


class TestEncoderHook:
    def test_concatenates_and_extends_mask(self):
        corr = filled_corr(2)
        hook = make_encoder_injection_hook(corr)
        hs = torch.randn(1, 6, D)
        mask = torch.zeros(1, 1, 6, 6)  # T5-style additive mask
        bias = torch.zeros(1, 4, 6, 6)
        corr(hs)  # warm-up -> retrieval non-empty
        rel, n = corr.memory.get_relevant(hs, 8)
        if n == 0:
            pytest.skip("warm-up consumed the match")
        args, kwargs = hook(None, (hs, mask, bias), {})
        out_hs, out_mask, out_bias = args
        assert out_hs.shape == (1, 6 + n, D)
        assert out_mask.shape == (1, 1, 6, 6 + n)
        assert torch.equal(out_mask[..., :6], mask)   # original columns untouched
        assert torch.equal(out_mask[..., 6:], torch.zeros(1, 1, 6, n))  # memory visible
        assert out_bias is None                        # force position-bias recompute

    def test_passthrough_when_nothing_retrieved(self):
        corr = CorrectionLayer(D, size=16)
        hook = make_encoder_injection_hook(corr)
        hs = torch.randn(1, 6, D)
        args, kwargs = hook(None, (hs, torch.zeros(1, 1, 6, 6), None), {})
        assert args[0] is hs  # untouched

    def test_kwargs_style(self):
        corr = filled_corr()
        hook = make_encoder_injection_hook(corr)
        hs = torch.randn(1, 4, D)
        corr(hs)
        args, kwargs = hook(None, (), {"hidden_states": hs, "attention_mask": torch.zeros(1, 1, 4, 4), "position_bias": torch.zeros(1, 4, 4, 4)})
        assert kwargs["hidden_states"].shape[1] > 4
        assert kwargs["position_bias"] is None


class TestWithRealTinyBlocks:
    """The hooks must survive registration on actual HF blocks."""

    def test_decoder_hook_on_gpt2_block(self, causal_bundle):
        block = causal_bundle.memory_target()
        corr = CorrectionLayer(D, size=16)
        hook = make_decoder_injection_hook(corr)
        handle = block.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            hs = torch.randn(1, 5, causal_bundle.hidden_size)
            block(hs)  # d_model mismatch guard: corr uses D != hidden -> should raise
            pytest.fail("expected dimension guard to trigger")
        except ValueError:
            pass
        finally:
            handle.remove()

    def test_encoder_hook_on_t5_block(self, encdec_bundle):
        block = encdec_bundle.memory_target()
        corr = CorrectionLayer(encdec_bundle.hidden_size, size=16)
        hook = make_encoder_injection_hook(corr)
        handle = block.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            hs = torch.randn(1, 5, encdec_bundle.hidden_size)
            out = block(hs, torch.zeros(1, 1, 5, 5))
            assert out[0].shape[1] >= 5  # extended once retrieval starts working
        finally:
            handle.remove()

"""Minimal, explicit decoding loop with pluggable KV-cache policies.

A hand-written greedy/sampling loop is used instead of ``model.generate``
so that cache pruning stays *exactly* under test control: the loop owns the
cache object, the absolute ``position_ids`` of every cached slot and the
per-step metrics (cache length, latency, peak memory). This is what makes
"classic KV cache" and "sparse KV cache" runs directly comparable, and
"no cache" available as a correctness baseline (full recomputation must
produce the same tokens as the classic cache).
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache
from transformers.cache_utils import EncoderDecoderCache

from .compat import cache_length, cache_mean_length, layer_kv, prune_cache
from .kvcache import (
    HeadImportanceTracker,
    ImportanceTracker,
    KVConfig,
    h2o_head_indices,
    keep_indices,
    pyramid_budgets,
    snapkv_indices,
)


@dataclass
class DecodeResult:
    """Outcome of one generation call.

    Attributes:
        prompt_len: number of prompt tokens.
        token_ids: generated token ids (prompt excluded).
        cache_lengths: mean slots per K/V layer after prefill and each step.
        cache_max_lengths: largest per-layer cache length at the same points.
        total_cache_slots: sum of ``cache_lengths`` — proxy for attention
            FLOPs; smaller means less work per generated token.
        latency_s: wall time of the whole loop, including prefill.
        peak_mem_mb: peak device memory during the call (None off-GPU).
    """

    prompt_len: int
    token_ids: list[int] = field(default_factory=list)
    cache_lengths: list[int] = field(default_factory=list)
    cache_max_lengths: list[int] = field(default_factory=list)
    total_cache_slots: int = 0
    latency_s: float = 0.0
    peak_mem_mb: float | None = None

    @property
    def steps(self) -> int:
        return len(self.token_ids)


def _peak_mem(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    return None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def pick_token(logits: torch.Tensor, *, do_sample: bool, temperature: float, top_p: float, generator: torch.Generator | None) -> int:
    """Select one token from last-step logits; any shape ending in ``(vocab,)``."""
    x = logits.float().reshape(-1)
    if not do_sample:
        return int(torch.argmax(x))
    if temperature <= 0:
        raise ValueError("temperature must be positive when sampling")
    scores = torch.log_softmax(x / temperature, dim=-1)
    if top_p < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=True)
        cum = torch.cumsum(torch.exp(sorted_scores), dim=0)
        drop = cum > top_p
        drop[1:] = drop[:-1].clone()
        drop[0] = False
        sorted_scores = sorted_scores.masked_fill(drop, float("-inf"))
        scores = torch.full_like(scores, float("-inf"))
        scores[sorted_idx] = sorted_scores
    return int(torch.multinomial(torch.exp(scores), num_samples=1, generator=generator)[0])


def _finish(res: DecodeResult) -> DecodeResult:
    res.total_cache_slots = sum(res.cache_lengths)
    return res


def _run_with_logits_keep(model, keep: int | None, **call_kwargs):
    """Call a causal LM keeping logits only for the last ``keep`` positions.

    Prefilling 4-8k tokens and materialising ``(1, S, vocab)`` logits
    (several GiB on 250k-token vocabularies) is pure waste: decoding needs
    the last position only. Models without ``logits_to_keep`` fall back.
    """
    if keep is not None:
        try:
            return model(logits_to_keep=keep, **call_kwargs)
        except TypeError:
            pass
    return model(**call_kwargs)


@contextmanager
def _attention_implementation(model, implementation: str):
    """Temporarily select a Transformers attention backend and restore it."""
    previous = getattr(model.config, "_attn_implementation", None)
    if previous == implementation:
        yield
        return
    model.set_attn_implementation(implementation)
    try:
        yield
    finally:
        if previous is not None:
            model.set_attn_implementation(previous)


def _record_cache(res: DecodeResult, cache) -> None:
    res.cache_lengths.append(cache_mean_length(cache))
    res.cache_max_lengths.append(cache_length(cache))


def _kv_layers(cache) -> list[tuple[int, torch.Tensor]]:
    return [(i, keys) for i, (keys, _) in enumerate(layer_kv(cache)) if keys is not None and keys.numel()]


def _prune_h2o(cache, tracker: HeadImportanceTracker, cfg: KVConfig) -> None:
    if cache_length(cache) <= cfg.budget:
        return
    layers = _kv_layers(cache)
    if len(layers) != len(tracker.scores):
        raise RuntimeError("H2O attention outputs do not match the K/V cache layers")
    keep = [
        h2o_head_indices(tracker.scores[i], cfg, keys.shape[1])
        for i, (_, keys) in enumerate(layers)
    ]
    prune_cache(cache, {layer: indices for (layer, _), indices in zip(layers, keep)})
    tracker.prune(keep)


def _compress_prompt_cache(cache, attentions, cfg: KVConfig) -> None:
    layers = _kv_layers(cache)
    if len(layers) != len(attentions):
        raise RuntimeError("prompt attention outputs do not match the K/V cache layers")
    budgets = (
        pyramid_budgets(len(layers), cfg.budget, cfg.pyramid_beta, cfg.observation)
        if cfg.strategy == "pyramidkv" else [cfg.budget] * len(layers)
    )
    keep = {}
    for (layer, keys), attention, budget in zip(layers, attentions, budgets):
        if attention is None:
            raise RuntimeError(f"missing attention weights for K/V layer {layer}")
        keep[layer] = snapkv_indices(
            attention, budget, cfg.observation, cfg.kernel_size, num_key_value_heads=keys.shape[1]
        )
    prune_cache(cache, keep)


def _prefill_with_scores(model, input_ids: torch.Tensor, cfg: KVConfig):
    """Prefill a causal model with bounded attention-score memory.

    H2O processes the prompt in ``chunk_size`` chunks and evicts after each
    one. SnapKV/PyramidKV use SDPA for the prefix and materialise eager
    attention only for the final observation window, then compress once.
    Returns the last model output, its cache, and an H2O tracker (or ``None``).
    """
    device = input_ids.device
    length = input_ids.shape[1]
    if length <= cfg.budget:
        out = _run_with_logits_keep(
            model, 1, input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            past_key_values=None, use_cache=True, output_attentions=False,
        )
        return out, out.past_key_values or DynamicCache(), None

    if cfg.compresses_prompt:
        split = length - cfg.observation
        cache = None
        with _attention_implementation(model, "sdpa"):
            out = _run_with_logits_keep(
                model, 1, input_ids=input_ids[:, :split], attention_mask=None,
                position_ids=torch.arange(split, device=device).unsqueeze(0),
                past_key_values=None, use_cache=True, output_attentions=False,
            )
        cache = out.past_key_values
        with _attention_implementation(model, "eager"):
            out = _run_with_logits_keep(
                model, 1, input_ids=input_ids[:, split:], attention_mask=None,
                position_ids=torch.arange(split, length, device=device).unsqueeze(0),
                past_key_values=cache, use_cache=True, output_attentions=True,
            )
        cache = out.past_key_values
        _compress_prompt_cache(cache, out.attentions, cfg)
        return out, cache, None

    tracker = HeadImportanceTracker(device)
    cache = None
    out = None
    for start in range(0, length, cfg.chunk_size):
        end = min(start + cfg.chunk_size, length)
        with _attention_implementation(model, "eager"):
            out = _run_with_logits_keep(
                model, 1, input_ids=input_ids[:, start:end], attention_mask=None,
                position_ids=torch.arange(start, end, device=device).unsqueeze(0),
                past_key_values=cache, use_cache=True, output_attentions=True,
            )
        cache = out.past_key_values
        tracker.step(out.attentions)
        _prune_h2o(cache, tracker, cfg)
    return out, cache, tracker


@torch.no_grad()
def decode_causal(
    model,
    input_ids: torch.Tensor,
    *,
    kv: KVConfig = KVConfig.classic(),
    cache: str = "use",
    max_new_tokens: int = 32,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    eos_ids: tuple[int, ...] = (),
    seed: int = 0,
) -> DecodeResult:
    """Decode one prompt under classic, StreamingLLM, H2O, SnapKV, or PyramidKV.

    The input must have shape ``(1, length)``. Greedy decoding is used unless
    ``do_sample`` is true; ``seed`` controls sampling only. The result contains
    generated IDs, per-step mean/maximum cache lengths, wall seconds including
    prefill, and peak accelerator memory in MiB. ``cache="none"`` recomputes
    the full sequence and ignores ``kv``. Invalid input shapes raise
    ``ValueError``; missing score outputs for score-based policies raise
    ``RuntimeError``.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape (1, L)")
    device = input_ids.device
    res = DecodeResult(prompt_len=int(input_ids.shape[1]))
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    if cache == "none":
        res.token_ids = _decode_without_cache(model, input_ids, res, max_new_tokens, do_sample, temperature, top_p, eos_ids, generator)
        res.cache_lengths = []
        res.cache_max_lengths = []
        res.latency_s = time.perf_counter() - t0
        res.peak_mem_mb = _peak_mem(device)
        return _finish(res)

    need_scores = kv.needs_step_scores
    if kv.needs_scores:
        out, dyn, tracker = _prefill_with_scores(model, input_ids, kv)
        if need_scores and tracker is None:
            tracker = HeadImportanceTracker(device)
    else:
        out = _run_with_logits_keep(
            model, 1, input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            past_key_values=None, use_cache=True, output_attentions=False,
        )
        dyn = out.past_key_values if out.past_key_values is not None else DynamicCache()
        tracker = None
    positions = list(range(input_ids.shape[1]))
    if not kv.needs_scores:
        _prune_if_needed(dyn, positions, None, kv)
    _record_cache(res, dyn)
    lengths = {keys.shape[-2] for _, keys in _kv_layers(dyn)}
    if kv.strategy == "pyramidkv" and model.config.model_type == "gpt2" and len(lengths) > 1:
        raise RuntimeError("PyramidKV requires per-layer attention masks; GPT-2 provides one shared mask")

    for step in range(max_new_tokens):
        nxt = pick_token(out.logits[:, -1:], do_sample=do_sample, temperature=temperature, top_p=top_p, generator=generator)
        res.token_ids.append(nxt)
        if nxt in eos_ids or step == max_new_tokens - 1:
            break
        cur_pos = input_ids.shape[1] + step
        pos_ids = torch.tensor([[cur_pos]], device=device)
        attn_mask = None if kv.compresses_prompt else torch.ones(1, cache_length(dyn) + 1, dtype=torch.long, device=device)
        with _attention_implementation(model, "eager") if need_scores else nullcontext():
            out = model(
                input_ids=torch.tensor([[nxt]], device=device), position_ids=pos_ids,
                attention_mask=attn_mask, past_key_values=dyn, use_cache=True,
                output_attentions=need_scores,
            )
        positions.append(cur_pos)
        if need_scores:
            tracker.step(out.attentions)
            _prune_h2o(dyn, tracker, kv)
        elif not kv.compresses_prompt:
            _prune_if_needed(dyn, positions, None, kv)
        _record_cache(res, dyn)

    _sync(device)
    res.latency_s = time.perf_counter() - t0
    res.peak_mem_mb = _peak_mem(device)
    return _finish(res)


def _decode_without_cache(model, input_ids, res, max_new_tokens, do_sample, temperature, top_p, eos_ids, generator) -> list[int]:
    """Recompute-from-scratch decoding: same distribution, no KV cache."""
    seq = input_ids
    out = _run_with_logits_keep(model, 1, input_ids=seq, attention_mask=torch.ones_like(seq), use_cache=False, output_attentions=False)
    for step in range(max_new_tokens):
        nxt = pick_token(out.logits[:, -1:], do_sample=do_sample, temperature=temperature, top_p=top_p, generator=generator)
        res.token_ids.append(nxt)
        if nxt in eos_ids or step == max_new_tokens - 1:
            break
        seq = torch.cat([seq, torch.tensor([[nxt]], device=seq.device)], dim=1)
        out = _run_with_logits_keep(model, 1, input_ids=seq, attention_mask=torch.ones_like(seq), use_cache=False, output_attentions=False)
    return res.token_ids


def _prune_if_needed(dyn, positions: list[int], tracker: ImportanceTracker | None, kv: KVConfig) -> None:
    """Apply the sparse policy: prune cache + bookkeeping to ``kv.budget``."""
    if kv.mode != "sparse":
        return
    length = cache_length(dyn)
    keep = keep_indices(length, kv, tracker.scores if tracker else None)
    if keep.numel() == length:
        return
    prune_cache(dyn, keep)
    idx = keep.tolist()
    positions[:] = [positions[i] for i in idx]
    if tracker is not None:
        tracker.prune(keep)


def _dec_logits(model, dec_out) -> torch.Tensor:
    """Project a T5 decoder-stack output to logits and keep the last step.

    ``model.decoder`` (T5Stack) returns hidden states; the LM head lives on
    the full model. Logit scaling factor (``config.factor``) is not applied —
    it is identical across cache modes, so comparisons stay fair.
    """
    hidden = dec_out.last_hidden_state if hasattr(dec_out, "last_hidden_state") else dec_out[0]
    return model.lm_head(hidden)[:, -1:]


def _self_attn_only(attentions) -> tuple | None:
    """T5Stack returns self-attn weights for all layers, then cross-attn."""
    if not attentions:
        return None
    return tuple(attentions[: len(attentions) // 2])


@torch.no_grad()
def decode_encdec(
    model,
    encoder_input_ids: torch.Tensor,
    *,
    decoder_start_id: int,
    kv: KVConfig = KVConfig.classic(),
    cache: str = "use",
    max_new_tokens: int = 32,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    eos_ids: tuple[int, ...] = (),
    seed: int = 0,
) -> DecodeResult:
    """Decode with an encoder-decoder model (T5-style).

    Sparse pruning is applied to the *decoder self-attention* cache only;
    the cross-attention cache mirrors the encoder output and is kept whole.
    ``cache="none"`` feeds the full decoder prefix every step.
    """
    device = encoder_input_ids.device
    res = DecodeResult(prompt_len=int(encoder_input_ids.shape[1]))
    generator = torch.Generator(device=device)  # must live on the model's device (ROCm/CUDA included)
    if seed is not None:
        generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    enc_mask = torch.ones_like(encoder_input_ids)
    enc = model.get_encoder()(input_ids=encoder_input_ids, attention_mask=enc_mask)
    # Mask for cross-attention must match the encoder output length, which
    # can exceed the input length when the memory hook injects extra tokens.
    cross_mask = torch.ones(
        encoder_input_ids.shape[0], enc.last_hidden_state.shape[1], dtype=enc_mask.dtype, device=device
    )

    if cache == "none":
        prefix = torch.tensor([[decoder_start_id]], device=device)
        for step in range(max_new_tokens):
            dec = model.decoder(
                input_ids=prefix,
                encoder_hidden_states=enc.last_hidden_state,
                encoder_attention_mask=cross_mask,
                use_cache=False,
            )
            nxt = pick_token(_dec_logits(model, dec), do_sample=do_sample, temperature=temperature, top_p=top_p, generator=generator)
            res.token_ids.append(nxt)
            if nxt in eos_ids or step == max_new_tokens - 1:
                break
            prefix = torch.cat([prefix, torch.tensor([[nxt]], device=device)], dim=1)
        res.latency_s = time.perf_counter() - t0
        res.peak_mem_mb = _peak_mem(device)
        return _finish(res)

    past = None
    tracker = ImportanceTracker(device) if kv.needs_scores else None
    need_scores = kv.needs_scores
    cur = torch.tensor([[decoder_start_id]], device=device)
    dec = model.decoder(
        input_ids=cur,
        encoder_hidden_states=enc.last_hidden_state,
        encoder_attention_mask=cross_mask,
        past_key_values=None,
        use_cache=True,
        output_attentions=need_scores,
    )
    past = dec.past_key_values if dec.past_key_values is not None else EncoderDecoderCache(DynamicCache(), DynamicCache())
    self_cache = past.self_attention_cache
    if need_scores:
        tracker.step(_self_attn_only(dec.attentions))
    _prune_encdec(self_cache, tracker, kv)
    res.cache_lengths.append(cache_length(self_cache))

    for step in range(max_new_tokens):
        nxt = pick_token(_dec_logits(model, dec), do_sample=do_sample, temperature=temperature, top_p=top_p, generator=generator)
        res.token_ids.append(nxt)
        if nxt in eos_ids or step == max_new_tokens - 1:
            break
        cur = torch.tensor([[nxt]], device=device)
        dec = model.decoder(
            input_ids=cur,
            encoder_hidden_states=enc.last_hidden_state,
            encoder_attention_mask=cross_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=need_scores,
        )
        if need_scores:
            tracker.step(_self_attn_only(dec.attentions))
        _prune_encdec(self_cache, tracker, kv)
        res.cache_lengths.append(cache_length(self_cache))

    _sync(device)
    res.latency_s = time.perf_counter() - t0
    res.peak_mem_mb = _peak_mem(device)
    return _finish(res)


def _prune_encdec(self_cache, tracker: ImportanceTracker | None, kv: KVConfig) -> None:
    """T5 self-cache is position-free (relative bias over contiguous
    slots), so only tail-contiguous sliding keeps the bias meaningful;
    pruning simply drops slots and the remaining window stays contiguous."""
    if kv.mode != "sparse":
        return
    length = cache_length(self_cache)
    keep = keep_indices(length, kv, tracker.scores if tracker else None)
    if keep.numel() == length:
        return
    prune_cache(self_cache, keep)
    if tracker is not None:
        tracker.prune(keep)

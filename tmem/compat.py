"""Compatibility shim over ``transformers`` cache layouts.

The internal structure of ``DynamicCache`` changed across transformers
versions (``key_cache``/``value_cache`` lists vs. ``layers[i].keys``).
This module normalises access so the rest of ``tmem`` (and its tests)
is version-independent, and additionally supports encoder-decoder caches
(``EncoderDecoderCache``).
"""

from __future__ import annotations

import torch


def self_cache(cache):
    """Return the growing self-attention cache of ``cache``.

    Accepts ``EncoderDecoderCache`` (returns its self-attention part),
    ``DynamicCache`` or ``None``.
    """
    inner = getattr(cache, "self_attention_cache", cache)
    return inner


def num_layers(cache) -> int:
    cache = self_cache(cache)
    if cache is None:
        return 0
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return len(layers)
    return len(getattr(cache, "key_cache", []))


def _read_slot(cache, layer_idx: int):
    layers = getattr(cache, "layers", None)
    if layers is not None:
        layer = layers[layer_idx]
        for names in (("keys", "values"), ("k", "v")):
            kv = tuple(getattr(layer, n, None) for n in names)
            if all(t is not None for t in kv):
                return kv
        return None, None  # non-KV layer (e.g. linear-attention state in Qwen3.5)
    kc = getattr(cache, "key_cache", None)
    if kc is not None and layer_idx < len(kc) and kc[layer_idx] is not None:
        return kc[layer_idx], cache.value_cache[layer_idx]
    return None, None


def _write_slot(cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        layer = layers[layer_idx]
        if hasattr(layer, "keys"):
            layer.keys, layer.values = keys, values
        else:
            layer.k, layer.v = keys, values
        return
    cache.key_cache[layer_idx] = keys
    cache.value_cache[layer_idx] = values


def layer_kv(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """``[(keys, values), ...]`` per layer of the self-attention cache."""
    cache = self_cache(cache)
    return [_read_slot(cache, i) for i in range(num_layers(cache))]


def cache_lengths(cache) -> list[int]:
    """Return the number of cached slots in every layer that stores K/V."""
    return [keys.shape[-2] for keys, _ in layer_kv(cache) if keys is not None and keys.numel()]


def cache_length(cache) -> int:
    """Return the largest per-layer cache length, or 0 for an empty cache."""
    return max(cache_lengths(cache), default=0)


def cache_mean_length(cache) -> int:
    """Return the rounded mean cache length across K/V layers.

    The mean is the attention-work unit for policies such as PyramidKV that
    intentionally assign different budgets to different layers. It is 0 when
    no layer holds K/V tensors.
    """
    lengths = cache_lengths(cache)
    return round(sum(lengths) / len(lengths)) if lengths else 0


def _gather_slots(tensors: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    if keep.ndim == 1:
        if keep.numel() and int(keep.amax()) >= tensors.shape[-2]:
            raise IndexError(f"prune index out of range: cache has {tensors.shape[-2]} slots")
        return tensors.index_select(-2, keep).contiguous()
    if keep.ndim != 2 or keep.shape[0] != tensors.shape[1]:
        raise ValueError(f"per-head keep must have shape (heads={tensors.shape[1]}, slots)")
    if keep.numel() and int(keep.amax()) >= tensors.shape[-2]:
        raise IndexError(f"prune index out of range: cache has {tensors.shape[-2]} slots")
    return torch.stack([
        tensors[:, head].index_select(-2, keep[head]) for head in range(tensors.shape[1])
    ], dim=1).contiguous()


def prune_cache(cache, keep: torch.Tensor | dict[int, torch.Tensor]) -> None:
    """Drop unselected K/V slots in place, optionally per layer and head.

    A 1-D tensor applies shared indices to every layer. A 2-D tensor has one
    row of sorted indices per K/V head. A dictionary maps cache-layer indices
    to either form; layers omitted from it remain unchanged. Indices refer to
    each layer's current slots. Out-of-range indices raise ``IndexError`` and
    incompatible head dimensions raise ``ValueError``.
    """
    inner = self_cache(cache)
    for i in range(num_layers(inner)):
        keys, values = _read_slot(inner, i)
        if keys is None or keys.shape[-2] == 0 or (isinstance(keep, dict) and i not in keep):
            continue
        local = keep[i] if isinstance(keep, dict) else keep
        local = local.to(keys.device)
        _write_slot(inner, i, _gather_slots(keys, local), _gather_slots(values, local))

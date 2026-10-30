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


def cache_length(cache) -> int:
    """Number of cached token slots (0 when empty; first layer that holds K/V)."""
    for keys, _ in layer_kv(cache):
        if keys is not None and keys.numel():
            return keys.shape[-2]
    return 0


def prune_cache(cache, keep: torch.Tensor) -> None:
    """Drop all cache slots whose index is not in ``keep`` (in place).

    ``keep`` is a 1-D index tensor sorted ascending, with values in
    ``[0, cache_length)``. Key/value tensors are gathered along the
    sequence dimension for every layer.
    """
    inner = self_cache(cache)
    for i in range(num_layers(inner)):
        keys, values = _read_slot(inner, i)
        if keys is None or keys.shape[-2] == 0:
            continue
        keep_local = keep.to(keys.device)
        if keep_local.numel() and int(keep_local.amax()) >= keys.shape[-2]:
            raise IndexError(f"prune index out of range: cache has {keys.shape[-2]} slots")
        _write_slot(
            inner,
            i,
            keys.index_select(-2, keep_local).contiguous(),
            values.index_select(-2, keep_local).contiguous(),
        )

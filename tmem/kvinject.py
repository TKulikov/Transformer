"""v4: activated facts enter the *attention* of mid-stack layers.

The author's original design — the hidden-state additive hook was a
simplified compromise. Intended behaviour:

* early layers: tokens are still lexical;
* **mid layers: tokens have converged into meanings — where memory belongs**;
* late layers: busy with the output task.

The bank of activated topic vectors joins the self-attention of a chosen
middle band of full-attention layers as extra **keys and values** (never
queries). Memory K/V skip RoPE on purpose: a remembered fact is
position-free relative to the dialogue stream. The 4-D additive mask gets
zero columns for memory slots (visible to every real token); causality
among real tokens is untouched, and memory never enters the KV-cache.

Mechanics: ``Qwen3_5Attention.forward`` is transparent
(Q/K/V projections -> RoPE -> cache update -> attention interface), so we
wrap it per instance mirroring the reference code plus injection after the
cache update.
"""

from __future__ import annotations

import torch

from .memory import CorrectionLayer

_NEG_INF = torch.finfo(torch.float32).min


class KVMemoryAttention:
    """Per-instance replacement for an attention module's forward."""

    def __init__(self, attn, corr: CorrectionLayer, max_slots: int = 32) -> None:
        self.attn = attn
        self.corr = corr
        self.max_slots = max_slots
        self.stats = {"steps": 0, "injected_slots": 0}

    def __call__(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        a = self.attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, a.head_dim)

        query_states, gate = torch.chunk(
            a.q_proj(hidden_states).view(*input_shape, -1, a.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = a.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = a.k_norm(a.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = a.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, a.layer_idx)

        # ---- activated facts as position-free extra keys/values ----
        self.stats["steps"] += 1
        with torch.no_grad():
            relevant, n = self.corr(hidden_states.detach().to(torch.float32))
        if n > 0:
            n = min(n, self.max_slots)
            mem = relevant[:, :n].to(hidden_states.dtype)
            mem_k = a.k_norm(a.k_proj(mem).view(1, n, -1, a.head_dim)).transpose(1, 2)
            mem_v = a.v_proj(mem).view(1, n, -1, a.head_dim).transpose(1, 2)
            key_states = torch.cat([key_states, mem_k], dim=-2)
            value_states = torch.cat([value_states, mem_v], dim=-2)
            attention_mask = _extend_mask(attention_mask, key_states.shape[-2] - n, n, key_states.dtype)
            self.stats["injected_slots"] += n

        attn_output, attn_weights = self._attention(a, query_states, key_states, value_states, attention_mask, kwargs)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = a.o_proj(attn_output)
        return attn_output, attn_weights

    @staticmethod
    def _attention(a, q, k, v, mask, kwargs):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        iface = ALL_ATTENTION_FUNCTIONS.get_interface(a.config._attn_implementation, M.eager_attention_forward)
        return iface(a, q, k, v, mask, dropout=0.0, scaling=a.scaling, **kwargs)


def _extend_mask(mask, real_keys: int, mem: int, dtype: torch.dtype):
    """Append ``mem`` visible columns to a 4-D additive mask.

    ``None`` mask means the fast path: single-token queries see everything
    (fine for memory), while multi-token prefill needs an explicit
    causal + memory mask built here.
    """
    if mask is None:
        return None  # q==1 decode: sdpa treats absent mask as fully visible
    b, h, s_q, s_k = mask.shape
    extra = torch.zeros(b, h, s_q, mem, dtype=dtype, device=mask.device)
    return torch.cat([mask, extra], dim=-1)


def install_kv_memory(model, corr: CorrectionLayer, layers: list[int] | None = None, max_slots: int = 32):
    """Wrap attention of the chosen layers with KV-memory injection.

    Args:
        model: Qwen3.5 text model (``Qwen3_5ForCausalLM``).
        corr: shared activation layer — the bank of live topics.
        layers: layer indices; default = middle band of full-attention
            layers (the semantic zone of the stack).
    Returns:
        list of :class:`KVMemoryAttention` wrappers (carry stats).
    """
    body = model.model if hasattr(model, "model") else model
    blocks = body.layers
    cfg = model.config
    layer_types = getattr(cfg, "layer_types", None) or getattr(
        getattr(cfg, "text_config", None), "layer_types", None
    )
    if layers is None:
        if layer_types:
            att = [i for i, t in enumerate(layer_types) if "full" in str(t)]
            n = len(att)
            layers = att[max(n // 4, 0): max(3 * n // 4, n // 4 + 1)] or att[1:2]
        else:
            layers = [len(blocks) // 2]
    wrappers = []
    for li in layers:
        attn = blocks[li].self_attn
        if not hasattr(attn, "_orig_forward"):
            attn._orig_forward = attn.forward
        wrap = KVMemoryAttention(attn, corr, max_slots)
        attn.forward = wrap
        wrappers.append(wrap)
    return wrappers


def uninstall_kv_memory(model) -> None:
    """Restore original attention forwards."""
    body = model.model if hasattr(model, "model") else model
    for block in body.layers:
        attn = getattr(block, "self_attn", None)
        if attn is not None and hasattr(attn, "_orig_forward"):
            attn.forward = attn._orig_forward
            del attn._orig_forward

"""Forward (pre-)hooks that inject the associative memory into HF models.

Two flavours, mirroring the notebook:

* :func:`make_encoder_injection_hook` — for encoder-decoder models (T5):
  the retrieved memory vectors are *concatenated* to the hidden states of
  an encoder block and the (4-D) attention mask is extended; the memory
  tokens act as extra keys/values that the rest of the encoder can attend
  to. A zeros extension means "visible" in T5's additive mask convention.
* :func:`make_decoder_injection_hook` — for decoder-only models (GPT-2,
  Qwen2): the mean of the relevant memory is added to hidden states with a
  small weight ``beta``.

Unlike the notebook, hooks are closures over the correction layer — no
module-level globals.
"""

from __future__ import annotations

from typing import Callable

import torch

from .memory import CorrectionLayer

Hook = Callable[..., tuple[tuple, dict] | None]


def _pick(args: tuple, kwargs: dict, name: str, position: int) -> torch.Tensor | None:
    value = kwargs.get(name)
    if value is None and len(args) > position and isinstance(args[position], torch.Tensor):
        value = args[position]
    return value


def make_decoder_injection_hook(corr: CorrectionLayer, beta: float = 0.25) -> Hook:
    """Additive memory influence for a decoder block's input hidden states.

    Register with ``module.register_forward_pre_hook(hook, with_kwargs=True)``.
    """

    def hook(module, args, kwargs):  # noqa: ANN001 - HF pre-hook signature
        hidden = _pick(args, kwargs, "hidden_states", 0)
        if hidden is None:
            return args, kwargs
        relevant, count = corr(hidden.to(torch.float32))
        if count > 0:
            influence = relevant.mean(dim=1, keepdim=True).to(hidden.dtype)
            hidden = hidden + beta * influence
        new_args = list(args)
        if new_args and new_args[0] is not hidden:
            new_args[0] = hidden
        elif not new_args:
            kwargs["hidden_states"] = hidden
        return tuple(new_args), kwargs

    return hook


def make_encoder_injection_hook(corr: CorrectionLayer) -> Hook:
    """Concat memory tokens to an encoder block's hidden states + mask."""

    def hook(module, args, kwargs):  # noqa: ANN001 - HF pre-hook signature
        hidden = _pick(args, kwargs, "hidden_states", 0)
        if hidden is None:
            return args, kwargs
        mask = _pick(args, kwargs, "attention_mask", 1)

        relevant, count = corr(hidden)
        if count == 0:
            return args, kwargs
        relevant = relevant.to(dtype=hidden.dtype, device=hidden.device)
        new_hidden = torch.cat([hidden, relevant], dim=1)

        def _extend(m: torch.Tensor | None) -> torch.Tensor | None:
            if m is None:
                return None
            extra = torch.zeros(*m.shape[:-1], count, dtype=m.dtype, device=m.device)
            return torch.cat([m, extra], dim=-1)

        if args:
            new_args = list(args)
            new_args[0] = new_hidden
            if len(new_args) > 1:
                new_args[1] = _extend(new_args[1])
                # Sequence length changed: force self-attention to recompute
                # the relative position bias instead of reusing a stale one.
                if len(new_args) > 2:
                    new_args[2] = None
            return tuple(new_args), kwargs
        kwargs["hidden_states"] = new_hidden
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = _extend(kwargs["attention_mask"])
        if "position_bias" in kwargs:
            kwargs["position_bias"] = None
        return args, kwargs

    return hook

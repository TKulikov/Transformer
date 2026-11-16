"""Model loading and the tiny stand-ins used by the test-suite.

``ModelBundle`` couples everything ``tmem`` needs from a model: the network,
the tokenizer, the kind (causal vs encoder-decoder) and the module the
memory hook is registered on (the same blocks the notebook picked:
GPT-2 ``h[6]``, Qwen2 ``layers[10]``, T5 ``encoder.block[3]``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import (
    DynamicCache,
    GPT2Config,
    GPT2LMHeadModel,
    T5Config,
    T5ForConditionalGeneration,
)

from .hooks import make_decoder_injection_hook, make_encoder_injection_hook
from .memory import CorrectionLayer

MODEL_REGISTRY = {
    "gpt2": {"kind": "causal", "hook_path": "transformer.h.6", "hidden": "n_embd"},
    "qwen2-1.5b": {
        "kind": "causal",
        "hook_path": "model.layers.10",
        "hidden": "hidden_size",
        "repo": "Qwen/Qwen2-1.5B",
    },
    "qwen35-4b": {
        "kind": "causal",
        "hook_path": None,  # resolved dynamically (hybrid layout)
        "hidden": "hidden_size",
        "repo": "Qwen/Qwen3.5-4B",
    },
    "t5-base": {"kind": "encdec", "hook_path": "encoder.block.3", "hidden": "d_model"},
}


@dataclass
class ModelBundle:
    """A loaded model plus everything needed to run memory/cache experiments."""

    name: str
    kind: str                      # "causal" | "encdec"
    model: nn.Module
    tokenizer: object
    device: torch.device

    @property
    def hidden_size(self) -> int:
        cfg = self.model.config
        for holder in (cfg, getattr(cfg, "text_config", None)):
            if holder is None:
                continue
            for attr in ("hidden_size", "n_embd", "d_model"):
                if hasattr(holder, attr) and isinstance(getattr(holder, attr), int):
                    return int(getattr(holder, attr))
        raise AttributeError("cannot determine hidden size")

    def memory_target(self) -> nn.Module:
        """The transformer block the memory hook is registered on.

        Uses the registry path when known; otherwise finds the deepest
        ``ModuleList`` named ``layers``/``h`` (works for hybrid Qwen3.5
        layouts too) and hooks a mid-stack block.
        """
        spec = MODEL_REGISTRY.get(self.name, {})
        path = spec.get("hook_path")
        if path is None:
            blocks = self._find_block_list()
            idx = min(11, len(blocks) - 1)  # prefer a full-attention layer if discoverable
            try:
                layer_types = getattr(self.model.config, "layer_types", None) or \
                    getattr(getattr(self.model.config, "text_config", None), "layer_types", None)
                if layer_types:
                    att = [i for i, t in enumerate(layer_types) if "full" in t or "attention" == t]
                    att = [a for a in att if a <= idx * 2] or att
                    if att:
                        idx = att[min(2, len(att) - 1)]
            except Exception:
                pass
            return blocks[idx]
        module: nn.Module = self.model
        for part in path.split("."):
            module = getattr(module, part) if not part.isdigit() else module[int(part)]
        return module

    def _find_block_list(self) -> nn.ModuleList:
        best: nn.ModuleList | None = None
        best_len = 0
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.ModuleList) and name.split(".")[-1] in ("layers", "h", "block"):
                if len(mod) > best_len:
                    best, best_len = mod, len(mod)
        if best is None or best_len == 0:
            raise ValueError("no transformer block list found in model")
        return best

    def bank_layer_indices(self) -> list[int]:
        """Indices of the layer band whose activations feed the latent bank.

        Same rule as :func:`tmem.kvinject.install_kv_memory`: the middle band
        of full-attention layers (semantic zone); fallback — the model middle.
        ``embed_utterance(layer="bank")`` keys RAG by these activations, so the
        retrieval space coincides with the injected CorrectionLayer space.
        """
        blocks = self._find_block_list()
        cfg = self.model.config
        layer_types = getattr(cfg, "layer_types", None) or getattr(
            getattr(cfg, "text_config", None), "layer_types", None
        )
        if layer_types:
            att = [i for i, t in enumerate(layer_types) if "full" in str(t)]
            n = len(att)
            return att[max(n // 4, 0): max(3 * n // 4, n // 4 + 1)] or att[1:2]
        return [len(blocks) // 2]

    def make_memory_hook(
        self,
        d_model: int | None = None,
        size: int = 512,
        beta: float = 0.25,
        window_crit: float = 10.0,
        retrieve_count: int = 256,
        crit: float = 0.8,
        max_to_new: float = 0.4,
        merge_k: float | None = None,
        decay: bool = True,
    ):
        """Create ``(hook, correction_layer)`` for this bundle's architecture."""
        corr = CorrectionLayer(
            d_model or self.hidden_size,
            size=size,
            window_crit=window_crit,
            retrieve_count=retrieve_count,
            crit=crit,
            max_to_new=max_to_new,
            merge_k=merge_k,
            decay=decay,
        )
        if self.kind == "encdec":
            return make_encoder_injection_hook(corr), corr
        return make_decoder_injection_hook(corr, beta=beta), corr

    def eos_ids(self) -> tuple[int, ...]:
        def flat(x):
            if isinstance(x, int):
                return [x]
            if isinstance(x, (list, tuple)):
                return [i for i in x if isinstance(i, int)]
            return []

        ids = set(flat(getattr(self.tokenizer, "eos_token_id", None))) | set(
            flat(getattr(self.model.config, "eos_token_id", None))
        )
        return tuple(sorted(ids))

    def to_chat(self, text: str) -> str:
        """Wrap a composed dialogue prompt into the model's chat format, if any.

        Reasoning-capable instruct models (Qwen3.5) are told to answer
        directly — our token budget measures the answer, not the chain of
        thought.
        """
        tmpl = getattr(self.tokenizer, "chat_template", None)
        if not tmpl:
            return text
        for kwargs in ({"enable_thinking": False}, {}):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, **kwargs
                )
            except (TypeError, ValueError, KeyError):
                continue
        return text

    def max_prompt_len(self, reserve: int = 0) -> int:
        """Longest prompt (in tokens) that still leaves ``reserve`` steps."""
        cfg = self.model.config
        limit = getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", None) or 512
        return max(int(limit) - reserve, 16)

    def decoder_start_id(self) -> int:
        return int(getattr(self.model.config, "decoder_start_token_id", None) or self.tokenizer.pad_token_id or 0)


def resolve(name: str) -> dict:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


def load_model(name: str, device: str | torch.device = "cpu", dtype: torch.dtype | str = "auto", eager: bool = False) -> ModelBundle:
    """Load a real checkpoint from the registry.

    Args:
        name: registry key (``gpt2``, ``qwen2-1.5b``, ``t5-base``).
        device: target device (``cpu``, ``cuda`` …).
        dtype: ``"auto"`` respects the checkpoint default; pass a
            ``torch.dtype`` to force it (fp32 is recommended for the
            associative memory experiments).
        eager: use the eager attention implementation — required when the
            sparse ``topk`` policy needs attention weights.
    """
    from transformers import AutoTokenizer

    spec = resolve(name)
    repo = spec.get("repo", name)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"dtype": dtype}
    if eager:
        kwargs["attn_implementation"] = "eager"
    if spec["kind"] == "causal":
        try:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
        except Exception:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(repo, **kwargs)
    else:
        from transformers import AutoModelForSeq2SeqLM

        model = AutoModelForSeq2SeqLM.from_pretrained(repo, **kwargs)
    dev = torch.device(device)
    model.to(dev)
    model.eval()
    return ModelBundle(name=name, kind=spec["kind"], model=model, tokenizer=tokenizer, device=dev)


# --------------------------------------------------------------------- tiny
class ByteTokenizer:
    """Deterministic fake tokenizer (utf-8 bytes) for offline unit tests."""

    vocab_size = 256
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text: str, truncation: bool = False, max_length: int | None = None, **kwargs):
        ids = list(text.encode("utf-8"))
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        data = [int(i) for i in ids if int(i) != self.eos_token_id] if skip_special_tokens else [int(i) for i in ids]
        return bytes(b % 256 for b in data).decode("utf-8", errors="ignore")


def tiny_causal(d_model: int = 64, layers: int = 2, vocab: int = 256, seed: int = 0) -> ModelBundle:
    """Tiny randomly-initialised GPT-2 (offline, CPU)."""
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=vocab,
        n_positions=512,
        n_embd=d_model,
        n_layer=layers,
        n_head=4,
        bos_token_id=ByteTokenizer.bos_token_id if hasattr(ByteTokenizer, "bos_token_id") else 0,
        eos_token_id=0,
    )
    cfg._attn_implementation = "eager"
    model = GPT2LMHeadModel(cfg)
    return ModelBundle(name="tiny-causal", kind="causal", model=model.eval(), tokenizer=ByteTokenizer(), device=torch.device("cpu"))


def tiny_encdec(d_model: int = 64, layers: int = 2, vocab: int = 256, seed: int = 0) -> ModelBundle:
    """Tiny randomly-initialised T5 (offline, CPU)."""
    torch.manual_seed(seed)
    cfg = T5Config(
        vocab_size=vocab,
        d_model=d_model,
        d_kv=16,
        d_ff=128,
        num_layers=layers,
        num_heads=4,
        decoder_start_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    cfg._attn_implementation = "eager"
    model = T5ForConditionalGeneration(cfg)
    return ModelBundle(name="tiny-encdec", kind="encdec", model=model.eval(), tokenizer=ByteTokenizer(), device=torch.device("cpu"))


def seed_forward_cache(bundle: ModelBundle) -> None:  # pragma: no cover - convenience helper
    """Warm-up helper kept for notebooks: run a one-token forward pass."""
    ids = torch.tensor([[1, 2, 3]])
    bundle.model(input_ids=ids, past_key_values=DynamicCache())
    torch.manual_seed(torch.initial_seed())

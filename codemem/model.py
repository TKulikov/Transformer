"""Model loading, embedding extraction and generation for codemem.

Fresh, minimal wrapper around Qwen3.5-4B (a hybrid linear/full-attention
multimodal checkpoint). No memory hooks — codemem reads hidden states for
embeddings and injects memory as text context, not as K/V. Loaded offline from
the local HF cache in bfloat16 on the ROCm card.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


@dataclass
class CodeModel:
    """A loaded generator plus embedding/generation helpers.

    Attributes:
        model: the transformer (eval mode, on ``device``).
        tokenizer: matching tokenizer (pad token defaults to eos).
        device: torch device the model lives on.
        embed_layer: default hidden-state layer for :meth:`embed`.
        pooling: default token pooling, "mean" or "last".
    """

    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    embed_layer: int = 15
    pooling: str = "mean"

    @property
    def hidden_size(self) -> int:
        """Language-model hidden width (reads text_config when nested)."""
        cfg = self.model.config
        for holder in (cfg, getattr(cfg, "text_config", None)):
            if holder is not None and isinstance(getattr(holder, "hidden_size", None), int):
                return int(holder.hidden_size)
        raise AttributeError("cannot determine hidden size")

    def to_chat(self, text: str) -> str:
        """Wrap text in the model's chat format, disabling chain-of-thought.

        Falls back to raw text if no chat template exists or it rejects the
        ``enable_thinking`` kwarg. Reasoning models are told to answer directly
        so token budgets measure the answer, not the thought trace.
        """
        tmpl = getattr(self.tokenizer, "chat_template", None)
        if not tmpl:
            return text
        for kwargs in ({"enable_thinking": False}, {}):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False, add_generation_prompt=True, **kwargs,
                )
            except (TypeError, ValueError, KeyError):
                continue
        return text

    @torch.no_grad()
    def embed(
        self,
        texts: list[str],
        layer: int | None = None,
        pooling: str | None = None,
    ) -> torch.Tensor:
        """Embed texts into hidden-state vectors [N, H] (float32, CPU).

        Args:
            texts: non-empty list of strings; each is embedded independently
                (batched with padding, mask-aware pooling).
            layer: hidden_states index (0 = embedding output, L = after layer L).
                Defaults to ``self.embed_layer``.
            pooling: "mean" (average over non-pad tokens) or "last" (final
                non-pad token). Defaults to ``self.pooling``.

        Returns:
            [N, H] float32 CPU tensor. Empty ``texts`` returns [0, H].

        Raises:
            IndexError: if ``layer`` exceeds the number of hidden states.
        """
        if not texts:
            return torch.zeros((0, self.hidden_size))
        layer = self.embed_layer if layer is None else layer
        pooling = self.pooling if pooling is None else pooling
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[layer].float()           # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
        if pooling == "mean":
            denom = mask.sum(dim=1).clamp_min(1.0)
            vec = (hs * mask).sum(dim=1) / denom
        elif pooling == "last":
            idx = enc["attention_mask"].sum(dim=1) - 1   # last non-pad position
            vec = hs[torch.arange(hs.shape[0], device=self.device), idx]
        else:
            raise ValueError(f"unknown pooling {pooling!r}")
        return vec.cpu()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 256, do_sample: bool = False) -> str:
        """Generate a completion for a raw prompt (wrapped via :meth:`to_chat`).

        Greedy by default (deterministic). Returns only the newly generated text
        (prompt stripped), with special tokens removed. ``max_new_tokens`` bounds
        the answer length.
        """
        text = self.to_chat(prompt)
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        eos = getattr(self.tokenizer, "eos_token_id", None)
        gen = self.model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=eos,
        )
        new = gen[0, enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip()


def load_model(
    repo: str = "Qwen/Qwen3.5-4B",
    device: str = "cuda",
    dtype: str = "bfloat16",
    embed_layer: int = 15,
    pooling: str = "mean",
) -> CodeModel:
    """Load a checkpoint offline and return a :class:`CodeModel`.

    Tries the causal-LM auto class first, then the image-text-to-text class
    (Qwen3.5-4B is a multimodal ``ForConditionalGeneration`` checkpoint). Sets
    pad token to eos when missing, moves to ``device`` and switches to eval.
    Requires the checkpoint in the local HF cache (run with HF_HUB_OFFLINE=1).
    """
    tok = AutoTokenizer.from_pretrained(repo)
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    kwargs = {"dtype": _dtype(dtype)}
    model = None
    for auto in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        try:
            import transformers
            model = getattr(transformers, auto).from_pretrained(repo, **kwargs)
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError(f"could not load {repo} with any auto class")
    dev = torch.device(device)
    model.to(dev)
    model.eval()
    return CodeModel(model=model, tokenizer=tok, device=dev, embed_layer=embed_layer, pooling=pooling)
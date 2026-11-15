"""v2 memory mechanics: EMA merging, utterance embeddings, RAG fact-bank.

The system's goal (per the author) is a fast dialogue *interlocutor*: keep
tone and topic alive from a bounded buffer while computing as little as
possible — not exact fact recall. This module upgrades the machinery:

* :class:`~tmem.memory.Memory` gains EMA merge ``a <- (1-k)a + k b``
  (bounded norm, 'a stays itself, b nudges') with merge threshold raised
  to 0.95 — see ``merge_k``;
* :class:`UtteranceEmbedder` turns a whole utterance into one vector
  (mean-pool of the model's last hidden states) — sentence granularity is
  what survives merging, unlike adjacent token states;
* :class:`RagBank` is the "sparse attention" analogue: exact fact *text*
  with upsert deduplication (an updated fact overwrites its twin instead of
  blending into it) and gist-conditioned retrieval
  ``q' = normalize(q + lambda * gist)`` — facts relevant to the current
  latent topic rank higher even without lexical overlap.
"""

from __future__ import annotations

import torch

from .memory import CorrectionLayer  # re-export for convenience


@torch.no_grad()
def embed_utterance(bundle, text: str, max_tokens: int = 128, layer: str = "last") -> torch.Tensor:
    """Mean-pooled hidden state of ``text`` at the chosen layer.

    layer: ``last`` — финальный слой (базовый RAG); ``mid`` — латентный слой
    смыслов (как ключ латентной памяти); ``mean`` — среднее по всем слоям.
    """
    ids = bundle.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)["input_ids"].to(bundle.device)
    model = bundle.model.get_encoder() if bundle.kind == "encdec" else bundle.model
    hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    if layer == "last":
        hidden = hs[-1]
    elif layer == "mid":
        hidden = hs[len(hs) // 2]
    elif layer == "mean":
        hidden = torch.stack(hs[1:]).mean(dim=0)
    else:
        raise ValueError(f"unknown embed layer: {layer}")
    return hidden.float().mean(dim=1).squeeze(0)  # (d,)


def model_last(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, output_hidden_states=True)
    return out.hidden_states[-1]


class RagBank:
    """Text-preserving fact bank with cosine-upsert dedup and gist retrieval.

    Args:
        crit: similarity at/above which a new utterance *replaces* the
            matching span (upsert) instead of appending a twin.
        capacity: max stored spans; oldest evicted when full (TTL-free —
            the bank is the long-term store, recency is handled by the
            dialogue buffer).
        centered: v3 — store and compare *deviations from the running mean*
            (the latent topic = the bank's own centroid). Raw embeddings of
            one model are anisotropic (mutual cos ~0.95+), which makes upsert
            collapse the bank and buries numeric facts; subtracting the mean
            removes the shared component, keeps the signal. Costs nothing at
            inference: the centroid is an aggregate of data we already embed.
    """

    def __init__(self, d_model: int, crit: float = 0.95, capacity: int = 256,
                 device: torch.device | str = "cpu", centered: bool = True) -> None:
        self.d_model = d_model
        self.crit = crit
        self.capacity = capacity
        self.device = torch.device(device)
        self.centered = centered
        self.reset()

    def reset(self) -> None:
        self.embeds = torch.zeros(0, self.d_model, device=self.device)
        self.raw_embeds = torch.zeros(0, self.d_model, device=self.device)
        self.texts: list[str] = []
        self.turns: list[int] = []
        self.mean = torch.zeros(self.d_model, device=self.device)
        self._n = 0
        self.stats = {"adds": 0, "upserts": 0, "evictions": 0}

    def _vec(self, emb: torch.Tensor) -> torch.Tensor:
        """Centre by the running topic and L2-normalise (raw if not centered)."""
        v = emb - self.mean if self.centered else emb
        return v / max(float(v.norm()), 1e-8)

    @torch.no_grad()
    def add(self, text: str, emb: torch.Tensor, turn: int = 0) -> None:
        # Questions are *lookup keys*, not content: storing them would let a
        # probe's embedding upsert-overwrite the very fact it asks about.
        if text.strip().endswith("?"):
            return
        emb = emb.float().to(self.device)
        # Centre by the topic centroid *of previously seen utterances* (the
        # first utterance has no mean to subtract), then grow the centroid.
        probe = self._vec(emb)
        if self.centered:
            self._n += 1
            self.mean = (self.mean * (self._n - 1) + emb) / self._n
        if self.embeds.numel():
            sims = self.embeds @ probe
            best = int(torch.argmax(sims))
            if float(sims[best]) >= self.crit:
                self.texts[best] = text          # upsert: updated fact wins
                self.embeds[best] = probe
                self.turns[best] = turn
                self.raw_embeds[best] = torch.nn.functional.normalize(emb, dim=0)
                self.stats["upserts"] += 1
                return
        if len(self.texts) >= self.capacity:
            self.embeds = self.embeds[1:]
            self.raw_embeds = self.raw_embeds[1:]
            self.texts = self.texts[1:]
            self.turns = self.turns[1:]
            self.stats["evictions"] += 1
        self.embeds = torch.cat([self.embeds, probe.view(1, -1)])
        self.raw_embeds = torch.cat([self.raw_embeds, torch.nn.functional.normalize(emb, dim=0).view(1, -1)])
        self.texts.append(text)
        self.turns.append(turn)
        self.stats["adds"] += 1

    @torch.no_grad()
    def retrieve(self, query: torch.Tensor, k: int = 4, gist: torch.Tensor | None = None,
                 lam: float = 0.0) -> list[tuple[str, int, float]]:
        """``[(text, turn, score)]`` ranked by cos(q', stored span).

        With ``centered`` the ranking is *deviation-vs-deviation* — the topic
        component is subtracted from both sides, which is exactly
        gist-conditioning for free. ``gist``/``lam`` keep the legacy
        memory-slot conditioning for comparison.
        """
        if self.embeds.numel() == 0:
            return []
        q = query.float().to(self.device)
        if gist is not None and lam > 0:
            q = q + lam * gist.float().to(self.device)
        q = self._vec(q)
        sims = self.embeds @ q
        top = torch.topk(sims, k=min(k, sims.numel()))
        return [(self.texts[i], self.turns[i], float(s)) for i, s in zip(top.indices.tolist(), top.values.tolist())]


def gist_of(corr: CorrectionLayer | None) -> torch.Tensor | None:
    """Mean vector over occupied memory slots — the latent-topic state."""
    if corr is None or corr.memory.count_element == 0:
        return None
    mask = corr.memory.occupied
    return corr.memory.context[0][mask].mean(dim=0)


def repetition_rate(text: str, n: int = 3) -> float:
    """Share of repeated word n-grams — catches degeneration loops ('I I I')."""
    words = text.lower().split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return round(1 - len(set(grams)) / len(grams), 3)


@torch.no_grad()
def tone_alignment(bundle, answer_emb: torch.Tensor, bank: "RagBank") -> float | None:
    """Max cosine of the answer against the dialogue's semantic bank.

    Proxies 'is the reply still about *this* conversation' — the interlocutor
    property the author optimizes for, as opposed to exact fact recall.
    Compared against *raw* (uncentred) span embeddings so the metric keeps
    its meaning across bank variants.
    """
    if bank.raw_embeds.numel() == 0:
        return None
    a = torch.nn.functional.normalize(answer_emb.float(), dim=0)
    return float((bank.raw_embeds @ a).max())

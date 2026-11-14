"""Generation-chunk memory: store what the model *produced*, not user facts.

Идея (заказ 13.11.2026): банк хранит чанки генерации (функции, абзацы),
а эмбеддинг чанка контекстуализируется самим чанком в потоке генерации —
не среднее последней активации по тексту, а пул скрытых состояний токенов
чанка, полученных в контексте породившего его промпта. Поиск по сходству
вытаскивает нужные куски собственного опыта агента (например, функции)
в момент, когда они нужны.

Отличия от :class:`tmem.semantic.RagBank` (факты пользователя):

* **что храним** — чанки ответов ассистента (:func:`chunk_text` режет код
  по ``def``/``class``, прозу по абзацам, с потолком по токенам);
* **ключ поиска** — активации слоёв самого блока (:func:`embed_block_key`,
  как hidden states в чат-версии): один запрос может породить тысячи строк,
  и ключ-запрос не различал бы куски одной генерации; контентный ключ
  позволяет вспоминать нужный кусок, а не всю генерацию. Legacy-режим
  ``chunk_key="request"`` оставлен для сравнения;
* **эмбеддинг значения** — :func:`embed_chunk` с режимами ``mean``
  (базовый, как ``embed_utterance``), ``span`` (пул только по токенам
  чанка в контексте), ``attn`` (взвешенный вниманием от последней позиции),
  ``late`` (per-token состояния для MaxSim, ColBERT-style);
* **центрирование** — и ключи, и значения центрируются своим бегущим
  средним (RagBank-style deviation scoring, снятие анизотропии);
* **upsert по умолчанию ВЫКЛЮЧЕН**: две похожие функции с разными
  именами не должны сливаться/перезаписываться по косинусу — для кода
  дедупликация по сходству опасна (в отличие от фактов-близнецов).

Ограничения (честно): контекст кодируется приближённо — граница span
определяется длиной токенизации префикса (BPE может «съехать» на стыке);
``attn`` использует внимание от последней позиции как прокси важности;
банк не верифицирует истинность собственных генераций (риск самоусиления
ошибок — нужен provenance/фильтр при агентном использовании).
"""

from __future__ import annotations

import re

import torch

from .semantic import model_last

_CODE_BOUNDARY = re.compile(r"(?m)^[ \t]*(?=(?:async[ \t]+def|def|class)[ \t])")
_PARAGRAPH = re.compile(r"\n\s*\n")


def chunk_text(text: str, max_tokens: int = 128, tokenizer=None) -> list[str]:
    """Split a generation into retrievable chunks.

    Code is split at every ``def``/``class`` boundary including indented
    methods (so a read file lands in memory as function-level pieces),
    prose at blank lines; any piece longer than ``max_tokens`` (when a
    tokenizer is given) is cut into consecutive token windows. Empty
    pieces are dropped.
    """
    text = text.strip()
    if not text:
        return []
    if _CODE_BOUNDARY.search(text):
        parts = [p.strip() for p in _CODE_BOUNDARY.split(text)]
    else:
        parts = [p.strip() for p in _PARAGRAPH.split(text)]
    parts = [p for p in parts if p]
    if tokenizer is None:
        return parts
    out: list[str] = []
    for p in parts:
        ids = tokenizer(p, return_tensors="pt", truncation=False)["input_ids"][0]
        if ids.numel() <= max_tokens:
            out.append(p)
            continue
        for i in range(0, ids.numel(), max_tokens):
            piece = tokenizer.decode(ids[i : i + max_tokens], skip_special_tokens=True).strip()
            if piece:
                out.append(piece)
    return out


def _forward_last(model, ids: torch.Tensor, need_attn: bool):
    """All hidden states (+ last-layer attentions) of a causal model or encoder.

    NOTE: SDPA/flash attention do not materialise attention weights —
    ``out.attentions`` is empty unless the model was loaded with
    ``eager=True``. In that case ``None`` is returned and the caller
    falls back to span-mean pooling.
    """
    out = model(input_ids=ids, output_hidden_states=True, output_attentions=need_attn)
    attn = out.attentions[-1] if (need_attn and out.attentions) else None
    return out.hidden_states, attn


def _pick_layer(hidden_states, layer: str) -> torch.Tensor:
    """Select the pooling source: last layer, latent (mid) layer, or layer mean.

    The latent (mid) layer is where tokens converge into meanings — the
    band where the v4 memory K/V is injected; contextualising values by
    its activations keeps the representation about *content*, not about
    the next-token output task.
    """
    if layer == "last":
        return hidden_states[-1]
    if layer == "mid":
        return hidden_states[len(hidden_states) // 2]
    if layer == "mean":
        return torch.stack(hidden_states[1:]).mean(dim=0)
    raise ValueError(f"unknown layer mode: {layer}")


@torch.no_grad()
def embed_chunk(
    bundle,
    chunk: str,
    context: str = "",
    mode: str = "span",
    max_tokens: int = 256,
    context_tokens: int = 512,
    layer: str = "last",
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Embed a generation chunk, contextualised by the chunk itself.

    Args:
        bundle: model bundle (causal or encdec).
        chunk: the generated piece to embed.
        context: the prompt that produced it (used by span/attn/late).
        mode: ``mean`` — standalone mean-pool (baseline); ``span`` — mean
            over the chunk's token span inside ``context + chunk``;
            ``attn`` — span pooled with attention weights received from the
            last position; ``late`` — span, plus per-token states returned
            for MaxSim retrieval.
        max_tokens: cap on chunk tokens; context_tokens: cap on prefix tokens.
        layer: pooling source — ``last`` (output layer), ``mid`` (the latent
            meaning layer, where v4 memory is injected) or ``mean`` (average
            over layers). Contextualising by the latent layer keeps the
            value about content rather than the next-token task.

    Returns:
        ``(pooled (d,), token_states (L, d) | None, n_tokens_consumed)``.
        ``token_states`` is non-None only for ``late``.
    """
    if mode not in ("mean", "span", "attn", "late"):
        raise ValueError(f"unknown chunk embed mode: {mode}")
    tok = bundle.tokenizer
    model = bundle.model.get_encoder() if bundle.kind == "encdec" else bundle.model

    if mode == "mean" or not context:
        ids = tok(chunk, return_tensors="pt", truncation=True, max_length=max_tokens)["input_ids"].to(bundle.device)
        hs, _ = _forward_last(model, ids, need_attn=False)
        hidden = _pick_layer(hs, layer)
        pooled = hidden.float().mean(dim=1).squeeze(0)
        tokens = hidden.float().squeeze(0) if mode == "late" else None
        return pooled, tokens, int(ids.shape[1])

    ctx_ids = tok(context, return_tensors="pt", truncation=True, max_length=context_tokens)["input_ids"]
    model_max = bundle.max_prompt_len(reserve=0)
    full = tok(context + chunk, return_tensors="pt", truncation=True,
               max_length=min(int(ctx_ids.shape[1]) + max_tokens, model_max))["input_ids"].to(bundle.device)
    start = min(int(ctx_ids.shape[1]), int(full.shape[1]))
    hs, attn = _forward_last(model, full, need_attn=(mode == "attn"))
    hidden = _pick_layer(hs, layer)
    span = hidden[:, start:, :]
    if span.shape[1] == 0:  # chunk vanished under truncation — fall back to whole
        span = hidden
    span = span.float()
    if mode == "attn" and attn is not None:
        # salience = attention the last position pays to each chunk token,
        # averaged over heads; renormalised over the span
        w = attn[0, :, -1, start:].float().mean(dim=0)
        w = w / torch.clamp(w.sum(), min=1e-8)
        pooled = (w[:, None] * span.squeeze(0)).sum(dim=0)
    else:
        pooled = span.mean(dim=1).squeeze(0)
    tokens = span.squeeze(0) if mode == "late" else None
    return pooled, tokens, int(full.shape[1])


@torch.no_grad()
def embed_block_key(bundle, chunk: str, layers: str = "mean",
                    max_tokens: int = 256) -> tuple[torch.Tensor, int]:
    """Content key: the block's own layer activations (chat-style hidden states).

    Один запрос может породить тысячи строк — ключ-запрос не различает куски
    одной генерации. Здесь ключ = пул скрытых состояний самого блока:
    ``last`` — последний слой, ``mid`` — средний (как hook в чат-версии),
    ``mean`` — среднее по всем слоям (активации слоёв, не только выхода).

    Returns:
        ``(key_emb (d,), n_tokens)``.
    """
    if layers not in ("last", "mid", "mean"):
        raise ValueError(f"unknown key layers mode: {layers}")
    ids = bundle.tokenizer(chunk, return_tensors="pt", truncation=True,
                           max_length=max_tokens)["input_ids"].to(bundle.device)
    model = bundle.model.get_encoder() if bundle.kind == "encdec" else bundle.model
    out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states
    if layers == "last":
        sel = [hs[-1]]
    elif layers == "mid":
        sel = [hs[len(hs) // 2]]
    else:
        sel = hs[1:]  # все слои, кроме входа (embedding output)
    pooled = torch.stack([h.float().mean(dim=1).squeeze(0) for h in sel]).mean(dim=0)
    return pooled, int(ids.shape[1])


class ChunkBank:
    """Bank of generation chunks: content keys, contextualised values.

    Each item is a ``(key, value)`` pair:

    * **key** — the block's own layer activations (:func:`embed_block_key`),
      so pieces of one huge generation are distinguishable by content;
      (legacy mode: the request embedding — ``chunk_key="request"``);
    * **value** — the chunk contextualised by the generation prompt
      (:func:`embed_chunk`, span/attn/late).

    Both sides are centred by their own running centroids and L2-normalised
    (RagBank-style deviation scoring: removes the shared anisotropic
    component of one model's embeddings). Raw normalised copies are kept
    for MaxSim / diagnostics.

    Args:
        upsert: replace the best-matching item when key similarity >= crit.
            Default False — for code, cosine dedup merges distinct functions.
        capacity: max items; oldest evicted when full.
    """

    def __init__(self, d_model: int, capacity: int = 256, centered: bool = True,
                 upsert: bool = False, crit: float = 0.95,
                 device: torch.device | str = "cpu") -> None:
        self.d_model = d_model
        self.capacity = capacity
        self.centered = centered
        self.upsert = upsert
        self.crit = crit
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.keys = torch.zeros(0, self.d_model, device=self.device)
        self.vals = torch.zeros(0, self.d_model, device=self.device)
        self.raw_keys = torch.zeros(0, self.d_model, device=self.device)
        self.raw_vals = torch.zeros(0, self.d_model, device=self.device)
        self.val_tokens: list[torch.Tensor | None] = []
        self.texts: list[str] = []
        self.turns: list[int] = []
        self.mean_key = torch.zeros(self.d_model, device=self.device)
        self.mean_val = torch.zeros(self.d_model, device=self.device)
        self._n = 0
        self.stats = {"adds": 0, "upserts": 0, "evictions": 0}

    def _vec(self, emb: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        v = emb - mean if self.centered else emb
        return v / torch.clamp(v.norm(), min=1e-8)

    @torch.no_grad()
    def add(self, text: str, key_emb: torch.Tensor, val_emb: torch.Tensor,
            turn: int = 0, val_tokens: torch.Tensor | None = None) -> None:
        """Store a chunk; key = block activations (or request), value = contextualised chunk."""
        if not text.strip():
            return
        key_emb = key_emb.float().to(self.device)
        val_emb = val_emb.float().to(self.device)
        probe_key = self._vec(key_emb, self.mean_key)
        probe_val = self._vec(val_emb, self.mean_val)
        if self.centered:
            self._n += 1
            self.mean_key = (self.mean_key * (self._n - 1) + key_emb) / self._n
            self.mean_val = (self.mean_val * (self._n - 1) + val_emb) / self._n
        raw_key = torch.nn.functional.normalize(key_emb, dim=0)
        raw_val = torch.nn.functional.normalize(val_emb, dim=0)
        toks = None
        if val_tokens is not None:
            toks = torch.nn.functional.normalize(val_tokens.float().to(self.device), dim=1)
        if self.upsert and self.keys.numel():
            sims = self.keys @ probe_key
            best = int(torch.argmax(sims))
            if float(sims[best]) >= self.crit:
                self.texts[best] = text
                self.keys[best] = probe_key
                self.vals[best] = probe_val
                self.raw_keys[best] = raw_key
                self.raw_vals[best] = raw_val
                self.val_tokens[best] = toks
                self.turns[best] = turn
                self.stats["upserts"] += 1
                return
        if len(self.texts) >= self.capacity:
            self.keys = self.keys[1:]
            self.vals = self.vals[1:]
            self.raw_keys = self.raw_keys[1:]
            self.raw_vals = self.raw_vals[1:]
            self.val_tokens = self.val_tokens[1:]
            self.texts = self.texts[1:]
            self.turns = self.turns[1:]
            self.stats["evictions"] += 1
        self.keys = torch.cat([self.keys, probe_key.view(1, -1)])
        self.vals = torch.cat([self.vals, probe_val.view(1, -1)])
        self.raw_keys = torch.cat([self.raw_keys, raw_key.view(1, -1)])
        self.raw_vals = torch.cat([self.raw_vals, raw_val.view(1, -1)])
        self.val_tokens.append(toks)
        self.texts.append(text)
        self.turns.append(turn)
        self.stats["adds"] += 1

    @torch.no_grad()
    def retrieve(self, query: torch.Tensor, k: int = 4, score: str = "key",
                 query_tokens: torch.Tensor | None = None) -> list[tuple[str, int, float]]:
        """Rank chunks against the current request (deviation-vs-deviation).

        Args:
            query: pooled embedding of the current request ``(d,)``.
            score: ``key`` — cos(request, block-activation key) [content lookup];
                ``value`` — cos(request, contextualised chunk);
                ``maxsim`` — ColBERT-style sum of per-query-token maxima
                against stored chunk tokens (needs ``query_tokens``; falls
                back to raw value cosine for items without token states).
        """
        if not self.texts:
            return []
        q = query.float().to(self.device)
        if score == "key":
            sims = self.keys @ self._vec(q, self.mean_key)
        elif score == "value":
            sims = self.vals @ self._vec(q, self.mean_val)
        elif score == "maxsim":
            if query_tokens is None:
                raise ValueError("maxsim needs query_tokens")
            qt = torch.nn.functional.normalize(query_tokens.float().to(self.device), dim=1)
            scores = []
            for i, toks in enumerate(self.val_tokens):
                if toks is None:
                    scores.append(float(self.raw_vals[i] @ torch.nn.functional.normalize(q, dim=0)))
                else:
                    scores.append(float((qt @ toks.T).max(dim=1).values.sum()))
            sims = torch.tensor(scores, device=self.device)
        else:
            raise ValueError(f"unknown chunk score mode: {score}")
        top = torch.topk(sims, k=min(k, sims.numel()))
        return [(self.texts[i], self.turns[i], float(s))
                for i, s in zip(top.indices.tolist(), top.values.tolist())]

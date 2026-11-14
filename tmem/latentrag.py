"""Латентный RAG: записи и запросы в одном латентном пространстве.

Вариация 3 (заказ 14.11). Отличия от старого RAG и от дерева:

* **Запись сохраняется ПОСЛЕ ответа** и сливает задачу с решением:
  ``r = center(norm(q_rec + beta * a))``, где ``q_rec`` — латентный mean
  запроса, ``a`` — латентный mean активаций ответа (один средний слой,
  без шума ранних слоёв и без «next-word» формата последнего).
* **Запрос контекстится латентной памятью с учётом TTL**:
  ``q' = center(norm(q + alpha * g_ttl))``, ``g_ttl`` — TTL-взвешенное
  среднее занятых слотов (живая тема, а не усреднённый шум).
* Ключ, значение и запрос читаются с ОДНОГО латентного слоя — единое
  пространство сравнения (устраняет мисматч old: key=all-layers,
  value=mid, query=final).

Слот-банк (:class:`tmem.memory.Memory`) наполняется латентными per-token
состояниями ответов и даёт TTL-динамику (refresh при извлечении, decay).
"""

from __future__ import annotations

import torch

from .chunkmem import embed_chunk
from .memory import Memory


def _norm(v: torch.Tensor) -> torch.Tensor:
    return v / torch.clamp(v.norm(), min=1e-8)


def latent_mean(bundle, text: str, max_tokens: int = 256) -> torch.Tensor:
    """Mean over tokens at the latent (mid) layer, standalone."""
    vec, _, _ = embed_chunk(bundle, text, mode="mean", layer="mid", max_tokens=max_tokens)
    return vec


class LatentRagBank:
    """Fused (request + answer) records, TTL-gist-conditioned latent queries.

    Args:
        beta: weight of the answer content in the stored record.
        alpha: weight of the TTL gist in the retrieval query (set per call).
        slot_size: capacity of the latent slot bank that produces the gist.
    """

    def __init__(self, d_model: int, beta: float = 1.0, capacity: int = 4096,
                 centered: bool = True, slot_size: int = 512,
                 device: torch.device | str = "cpu") -> None:
        self.d_model = d_model
        self.beta = beta
        self.capacity = capacity
        self.centered = centered
        self.device = torch.device(device)
        self.slots = Memory(d_model, size=slot_size, merge_k=0.25, device=device)
        self.reset()

    def reset(self) -> None:
        self.embeds = torch.zeros(0, self.d_model, device=self.device)
        self.texts: list[str] = []
        self.requests: list[str] = []
        self.turns: list[int] = []
        self.mean = torch.zeros(self.d_model, device=self.device)
        self._n = 0
        self.stats = {"adds": 0, "evictions": 0, "gist_used": 0}

    def _center(self, v: torch.Tensor) -> torch.Tensor:
        return _norm(v - self.mean) if self.centered else _norm(v)

    @torch.no_grad()
    def add(self, bundle, request: str, answer: str, context: str = "",
            turn: int = 0, max_tokens: int = 256) -> None:
        """Store a fused record after the answer; feed the latent slot bank."""
        if not answer.strip():
            return
        q_rec = latent_mean(bundle, request, max_tokens)
        a, a_toks, _ = embed_chunk(bundle, answer, context=context, mode="late",
                                   layer="mid", max_tokens=max_tokens)
        if a_toks is not None and a_toks.shape[0] > 0:
            # слот-банк здесь только для TTL-gist (не K/V-инъекция), поэтому
            # activity-gate смягчаем: код-активации иначе отсекаются порогом,
            # настроенным под диалоговые состояния, и gist остаётся пустым
            self.slots.update(a_toks.to(self.device), crit=0.95,
                              max_to_new=1.0, window_crit=-1e9)
        raw = _norm(q_rec + self.beta * a)
        probe = self._center(raw)
        if self.centered:
            self._n += 1
            self.mean = (self.mean * (self._n - 1) + raw) / self._n
        if len(self.texts) >= self.capacity:
            self.embeds = self.embeds[1:]
            self.texts = self.texts[1:]
            self.requests = self.requests[1:]
            self.turns = self.turns[1:]
            self.stats["evictions"] += 1
        self.embeds = torch.cat([self.embeds, probe.view(1, -1)])
        self.texts.append(answer)
        self.requests.append(request)
        self.turns.append(turn)
        self.stats["adds"] += 1

    @torch.no_grad()
    def retrieve(self, bundle, query: str, k: int = 4, alpha: float = 0.5,
                 max_tokens: int = 256) -> list[tuple[str, int, float]]:
        """Latent query conditioned by the TTL-weighted live-topic gist."""
        if not self.texts:
            return []
        q = latent_mean(bundle, query, max_tokens)
        self.slots.get_relevant(q.view(1, 1, -1))  # refresh relevant TTLs, decay all
        g = self.slots.ttl_gist()
        if g is not None and alpha > 0:
            q = q + alpha * g.to(q.device)
            self.stats["gist_used"] += 1
        qp = self._center(_norm(q))
        sims = self.embeds @ qp
        top = torch.topk(sims, k=min(k, sims.numel()))
        return [(self.texts[i], self.turns[i], float(s))
                for i, s in zip(top.indices.tolist(), top.values.tolist())]

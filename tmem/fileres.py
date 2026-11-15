"""Файл-резидуальный банк: двухстадийный поиск с разреживанием внутри файла.

Заказ 15.11 (вечер): «сильнее раскидать эмбеддинги». Наблюдение ретеста
(search_retest): mid-ключи похожих функций схлопываются — top-k забивается
клонами (5 одинаковых def summarize), бюджет заметок теряется. Лечение:

* ключ функции = контекстуализированный латентный эмбеддинг: span тела в
  контексте окружения (класс/файл) на mid-слое (опционально fused с
  сигнатурой: norm(q_rec + beta·a));
* **вычитается бегущее среднее ФАЙЛА** (центроид его функций) — общая часть
  файла уходит, остаётся отклонение → поиск внутри файла разреженный;
* запрос = латентный mean текущей задачи; stage 1 — выбор файла по
  центроиду (центрированному глобально, иначе анизотропия сливает файлы),
  stage 2 — запрос резидуализуется тем же центроидом (q − c_f) и плоский
  поиск по residual-ключам внутри выбранных файлов.

Отличие от HierLatentBank: там residual от ключа РОДИТЕЛЯ (заголовок класса)
и жадный спуск / shallow по лучшему члену; здесь — от центроида файла, и
файл выбирается прямым матчем запроса, и запрос вычитает общую часть файла.

Ингестия хранит компоненты (_qrec, _a) → rekey(beta) пересобирает ключи без
embed-форвардов (свип beta дешёвый).
"""

from __future__ import annotations

import torch

from .chunkmem import embed_chunk
from .hiermem import CodeNode, request_proxy
from .latentrag import latent_mean


def _norm(v: torch.Tensor) -> torch.Tensor:
    return v / torch.clamp(v.norm(), min=1e-8)


class FileResidualBank:
    """Двухстадийный поиск: файл по центроиду → плоский residual-поиск внутри.

    Args:
        nodes: дерево проекта (build_project_tree), родители раньше детей.
        beta: вес контекстного латента в fused-ключе norm(q_rec + beta·a);
            beta=0 — ключ = чистый контекстуализированный эмбеддинг тела.
        k_files: сколько файлов брать на stage 1 (ошибка stage 1 фатальна —
            k_files>1 страхует).
    """

    def __init__(self, nodes: list[CodeNode], beta: float = 0.0,
                 device: torch.device | str = "cpu") -> None:
        self.nodes = nodes
        self.beta = beta
        self.device = torch.device(device)
        self.file_of: dict[int, int] = {}
        self.funcs: list[CodeNode] = []
        self.centroids: dict[int, torch.Tensor] = {}
        self.file_keys: dict[int, torch.Tensor] = {}
        self.mean_c: torch.Tensor | None = None
        self.stats = {"functions": 0, "files": 0}
        for n in nodes:
            if n.kind != "function":
                continue
            p = n
            while p.parent is not None and p.kind != "file":
                p = nodes[p.parent]
            self.file_of[n.nid] = p.nid
            self.funcs.append(n)

    @torch.no_grad()
    def ingest(self, bundle, max_tokens: int = 256) -> None:
        """Эмбеддинг каждой функции в контексте окружения на латентном слое."""
        for n in self.funcs:
            parent = self.nodes[n.parent] if n.parent is not None else None
            ctx = parent.text if parent is not None else ""
            a, _, _ = embed_chunk(bundle, n.text, context=ctx, mode="span",
                                  layer="mid", max_tokens=max_tokens)
            n._qrec = latent_mean(bundle, request_proxy(n), max_tokens)
            n._a = a.float().to(self.device)
            self.stats["functions"] += 1
        self.rekey(self.beta)

    def rekey(self, beta: float) -> None:
        """Пересобрать ключи/центроиды/residual из компонент (без форвардов)."""
        self.beta = beta
        sums: dict[int, torch.Tensor] = {}
        counts: dict[int, int] = {}
        for n in self.funcs:
            raw = _norm(n._a if beta == 0 else n._qrec + beta * n._a)
            n.key_raw = raw.to(self.device)
            fid = self.file_of[n.nid]
            sums[fid] = sums.get(fid, torch.zeros_like(raw)) + raw
            counts[fid] = counts.get(fid, 0) + 1
        self.centroids = {fid: s / counts[fid] for fid, s in sums.items()}
        self.stats["files"] = len(self.centroids)
        # глобальное центрирование центроидов для stage 1 (снятие анизотропии)
        self.mean_c = torch.stack(list(self.centroids.values())).mean(dim=0)
        self.file_keys = {fid: _norm(c - self.mean_c) for fid, c in self.centroids.items()}
        for n in self.funcs:
            n.key_res = _norm(n.key_raw - self.centroids[self.file_of[n.nid]])

    @torch.no_grad()
    def retrieve(self, bundle, query: str, k_files: int = 2, k: int = 6,
                 max_tokens: int = 256) -> tuple[list[tuple[CodeNode, float, str]], list[str]]:
        """Stage 1: файл по центроиду; stage 2: плоский residual-поиск внутри.

        Returns:
            (top-k функций [(node, score, "function")], имена выбранных файлов).
        """
        q = latent_mean(bundle, query, max_tokens).to(self.device)
        q1 = _norm(q - self.mean_c)
        file_scores = sorted(((float(fk @ q1), fid) for fid, fk in self.file_keys.items()),
                             reverse=True)
        picked = [fid for _, fid in file_scores[:k_files]]
        names = [self.nodes[fid].name for fid in picked]
        scored: list[tuple[float, CodeNode]] = []
        for fid in picked:
            q_res = _norm(q - self.centroids[fid])
            for n in self.funcs:
                if self.file_of[n.nid] == fid:
                    scored.append((float(n.key_res @ q_res), n))
        scored.sort(key=lambda t: -t[0])
        return [(n, s, "function") for s, n in scored[:k]], names

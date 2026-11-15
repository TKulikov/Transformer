"""Семантическое дерево связей: модель сама собирает группы похожих функций.

Заказ 15.11 (ночь): деление по файлам ненадёжно (fileres: центроид grab-bag
utils размыт, stage 1 слеп к хелперам). Вместо файловой структуры — дерево,
которое модель строит САМА по смыслу:

* обработка функции → эмбеддинг (ключ) + текст (значение); сначала просто
  набрасываем в пространство знаний;
* перед вставкой следующей функции смотрим сходство с существующими
  группами: достаточно похожа (cos >= tau) → вступает в группу, иначе
  создаёт новую (онлайн leader-кластеризация, порядок = порядок ингестии);
* из ключей группы вычитается её среднее → residual-ключи листьев
  (разреживание внутри группы, как в fileres, но группы адекватные);
* поиск: запрос → top групп по центроидам → запрос вычитает центроид
  группы → плоский residual-поиск по листьям. Одиночки скорятся абсолютным
  deviation-ключом (residual для них вырожден).

Кластеризация в deviation-пространстве (raw − глобальное среднее, затем
norm): raw mid-эмбеддинги анизотропны (все cos>0.8), порог на них не
работает. Схлопывание похожих функций — здесь СИГНАЛ группировки, а не баг.

Ингестия хранит raw-векторы в двух пространствах (mid контекстный и
mean-слоёв контентный) → recluster(space, tau) пересобирает дерево без
embed-форвардов (свип порога/пространства дешёвый).

Глубокое дерево (заказ 15.11, ночь): группа — тоже узел с эмбеддингом
(центроидом). Если два центроида похожи (cos >= tau), они объединяются под
общим родителем — то же правило рекурсивно на каждом уровне → дендрограмма
произвольной глубины (max_depth). Поиск по глубокому дереву — beam-спуск с
пере-резидуализацией запроса на каждом узле (q_local = norm(q − центроид)),
как в HierLatentBank.descent, но дерево семантическое, а не файловое.

Ограничения: онлайн-порядок влияет на состав групп; residual почти
идентичных клонов может усиливать шум; жадный спуск может застревать в
ветке (страховка — beam>1 и flat-режим по группам первого уровня).
"""

from __future__ import annotations

import torch

from .chunkmem import embed_block_key, embed_chunk
from .hiermem import CodeNode
from .latentrag import latent_mean


def _norm(v: torch.Tensor) -> torch.Tensor:
    return v / torch.clamp(v.norm(), min=1e-8)


class SemTreeBank:
    """Онлайн-дерево семантических групп с residual-ключами листьев.

    Args:
        nodes: дерево проекта (build_project_tree); берутся только функции.
        space: пространство ключей — ``mid`` (контекстный латент) или
            ``mean`` (контентный mean всех слоёв, дискриминативнее).
        tau: порог сходства (cos в deviation-пространстве) для вступления
            в группу; выше → больше одиночек (ближе к flat).
    """

    def __init__(self, nodes: list[CodeNode], space: str = "mid", tau: float = 0.6,
                 device: torch.device | str = "cpu") -> None:
        self.nodes = nodes
        self.funcs = [n for n in nodes if n.kind == "function"]
        self.space = space
        self.tau = tau
        self.device = torch.device(device)
        self.raw: dict[str, torch.Tensor] = {}   # nid -> {"mid": (d,), "mean": (d,)}
        self.gmean: dict[str, torch.Tensor] = {}
        self.dev: torch.Tensor | None = None     # (n, d) нормированные отклонения
        self.clusters: list[dict] = []           # {"members": [idx], "centroid": (d,)}
        self.residuals: torch.Tensor | None = None
        self.leaf_res: torch.Tensor | None = None
        self.tree: dict | None = None
        self.max_depth = 1
        self.stats = {"functions": 0}

    @torch.no_grad()
    def ingest(self, bundle, max_tokens: int = 256) -> None:
        """Эмбеддинг каждой функции: mid-span в контексте окружения + mean-ключ."""
        mids, means = [], []
        for n in self.funcs:
            parent = self.nodes[n.parent] if n.parent is not None else None
            ctx = parent.text if parent is not None else ""
            a, _, _ = embed_chunk(bundle, n.text, context=ctx, mode="span",
                                  layer="mid", max_tokens=max_tokens)
            mk, _ = embed_block_key(bundle, n.text, layers="mean", max_tokens=max_tokens)
            mids.append(a.float().to(self.device))
            means.append(mk.float().to(self.device))
            self.stats["functions"] += 1
        self.raw = {"mid": torch.stack(mids), "mean": torch.stack(means)}
        self.recluster(self.space, self.tau)

    def recluster(self, space: str, tau: float, max_depth: int = 1) -> None:
        """Пересобрать дерево из хранящихся векторов (без embed-форвардов).

        max_depth=1 — плоские группы (как раньше); >1 — рекурсивное
        объединение похожих центроидов в более глубокое дерево.
        """
        self.space, self.tau, self.max_depth = space, tau, max_depth
        vecs = self.raw[space]
        g = vecs.mean(dim=0)
        self.gmean[space] = g
        dev = torch.stack([_norm(v - g) for v in vecs])
        self.dev = dev
        clusters = self._leader(dev, tau)
        self.clusters = clusters
        res = torch.zeros_like(dev)
        for cl in clusters:
            if len(cl["members"]) > 1:
                for i in cl["members"]:
                    res[i] = _norm(dev[i] - cl["centroid"])
        self.residuals = res
        self.leaf_res = res
        self.tree = self._build_tree(clusters, tau, max_depth)
        self.stats.update({"clusters": len(clusters),
                           "singletons": sum(1 for c in clusters if len(c["members"]) == 1),
                           "max_cluster": max(len(c["members"]) for c in clusters),
                           "depth": self._depth(self.tree), "nodes": self._count(self.tree)})

    @staticmethod
    def _leader(vecs: torch.Tensor, tau: float) -> list[dict]:
        """Онлайн leader-кластеризация набора векторов (порядок = порядок строк)."""
        clusters: list[dict] = []
        centroids: list[torch.Tensor] = []
        for i, d in enumerate(vecs):
            if centroids:
                sims = torch.stack([_norm(c) @ d for c in centroids])
                best = int(torch.argmax(sims))
                if float(sims[best]) >= tau:
                    cl = clusters[best]
                    cl["members"].append(i)
                    cl["sum"] += d
                    cl["centroid"] = cl["sum"] / len(cl["members"])
                    centroids[best] = cl["centroid"]
                    continue
            clusters.append({"members": [i], "sum": d.clone(), "centroid": d.clone()})
            centroids.append(d.clone())
        return clusters

    def _build_tree(self, clusters: list[dict], tau: float, max_depth: int) -> dict:
        """Рекурсивно объединяет похожие центроиды: группа — тоже узел."""
        level = [{"centroid": cl["centroid"], "leaves": list(cl["members"]),
                  "children": [], "res": None, "level": 1} for cl in clusters]
        depth = 1
        while depth < max_depth and len(level) > 1:
            cent = torch.stack([n["centroid"] for n in level])
            groups = self._leader(cent, tau)
            if len(groups) == len(level):  # ничего не слилось — дальше бессмысленно
                break
            parents = []
            for gr in groups:
                kids = [level[i] for i in gr["members"]]
                node = {"centroid": gr["centroid"], "leaves": [], "children": kids,
                        "res": None, "level": depth + 1}
                for ch in kids:
                    ch["res"] = _norm(ch["centroid"] - node["centroid"])
                parents.append(node)
            level = parents
            depth += 1
        root_cent = torch.stack([n["centroid"] for n in level]).mean(dim=0)
        root = {"centroid": root_cent, "leaves": [], "children": level,
                "res": None, "level": depth + 1}
        for ch in level:
            ch["res"] = _norm(ch["centroid"] - root_cent)
        return root

    @staticmethod
    def _depth(node: dict) -> int:
        return 1 + max((SemTreeBank._depth(c) for c in node["children"]), default=0)

    @staticmethod
    def _count(node: dict) -> int:
        return 1 + sum(SemTreeBank._count(c) for c in node["children"])

    def query_vec(self, bundle, query: str, max_tokens: int = 256) -> torch.Tensor:
        """Запрос в том же пространстве, что и ключи (mid или mean)."""
        if self.space == "mid":
            q = latent_mean(bundle, query, max_tokens)
        else:
            q, _ = embed_block_key(bundle, query, layers="mean", max_tokens=max_tokens)
        return _norm(q.float().to(self.device) - self.gmean[self.space])

    @torch.no_grad()
    def retrieve(self, bundle, query: str, k_clusters: int = 3, k: int = 6,
                 max_tokens: int = 256, mode: str = "flat"
                 ) -> tuple[list[tuple[CodeNode, float, str]], list[dict]]:
        """Поиск по дереву: flat (top групп 1-го уровня) или descent (beam-спуск).

        flat: запрос → top-k_clusters групп по центроиду → residual-листья
        внутри (одиночки — абсолютным deviation-ключом). Игнорирует уровни
        выше первого.

        descent: beam-спуск от корня; на каждом узле запрос пере-резидуализуется
        (q_local = norm(q − центроид_узла)), дети ранжируются по residual-ключам,
        листья собираются с накопленным path-score. Использует всю глубину.

        Returns:
            (top-k [(node, score, "leaf")], диагностика выбранных групп 1-го
            уровня [{"cluster": id, "size": n, "members": [имена top-3]}]).
        """
        dq = self.query_vec(bundle, query, max_tokens)
        if mode == "descent":
            scored = self._descent(dq, k_clusters, k)
        else:
            scored = self._flat(dq, k_clusters, k)
        results = [(self.funcs[i], s, "leaf") for s, i in scored]
        owner = {}
        for j, cl in enumerate(self.clusters):
            for i in cl["members"]:
                owner[i] = j
        seen, diag = set(), []
        for _, i in scored:
            j = owner.get(i)
            if j is not None and j not in seen:
                seen.add(j)
                cl = self.clusters[j]
                diag.append({"cluster": j, "size": len(cl["members"]),
                             "members": [self.funcs[x].name for x in cl["members"][:3]]})
        return results, diag

    def _flat(self, dq: torch.Tensor, k_clusters: int, k: int) -> list[tuple[float, int]]:
        scored_cl = sorted(((float(_norm(cl["centroid"]) @ dq), j)
                            for j, cl in enumerate(self.clusters)), reverse=True)
        scored: list[tuple[float, int]] = []
        for _, j in scored_cl[:k_clusters]:
            cl = self.clusters[j]
            if len(cl["members"]) == 1:
                i = cl["members"][0]
                scored.append((float(self.dev[i] @ dq), i))
            else:
                q_res = _norm(dq - cl["centroid"])
                for i in cl["members"]:
                    scored.append((float(self.residuals[i] @ q_res), i))
        scored.sort(key=lambda t: -t[0])
        return scored[:k]

    def _descent(self, dq: torch.Tensor, beam: int, k: int) -> list[tuple[float, int]]:
        beam_nodes = [(self.tree, 0.0)]
        cand: dict[int, float] = {}
        for _ in range(self.max_depth + 1):
            nxt: list[tuple[dict, float]] = []
            for node, ps in beam_nodes:
                q_local = _norm(dq - node["centroid"])
                for i in node["leaves"]:
                    s = ps + float(self.leaf_res[i] @ q_local)
                    if i not in cand or s > cand[i]:
                        cand[i] = s
                kids = sorted(((float(ch["res"] @ q_local), ch) for ch in node["children"]),
                              key=lambda t: -t[0])
                for s, ch in kids[:beam]:
                    nxt.append((ch, ps + s))
            nxt.sort(key=lambda t: -t[1])
            beam_nodes = nxt[:beam]
            if not beam_nodes:
                break
        return [(s, i) for i, s in sorted(cand.items(), key=lambda t: -t[1])[:k]]

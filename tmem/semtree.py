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

Ограничения v1: два уровня (группа→листья); онлайн-порядок влияет на
состав групп; residual почти идентичных клонов может усиливать шум.
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

    def recluster(self, space: str, tau: float) -> None:
        """Пересобрать дерево из хранящихся векторов (без embed-форвардов)."""
        self.space, self.tau = space, tau
        vecs = self.raw[space]
        g = vecs.mean(dim=0)
        self.gmean[space] = g
        dev = torch.stack([_norm(v - g) for v in vecs])
        self.dev = dev
        clusters: list[dict] = []
        centroids: list[torch.Tensor] = []
        for i, d in enumerate(dev):
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
        self.clusters = clusters
        res = torch.zeros_like(dev)
        for cl in clusters:
            if len(cl["members"]) > 1:
                for i in cl["members"]:
                    res[i] = _norm(dev[i] - cl["centroid"])
        self.residuals = res
        self.stats.update({"clusters": len(clusters),
                           "singletons": sum(1 for c in clusters if len(c["members"]) == 1),
                           "max_cluster": max(len(c["members"]) for c in clusters)})

    def query_vec(self, bundle, query: str, max_tokens: int = 256) -> torch.Tensor:
        """Запрос в том же пространстве, что и ключи (mid или mean)."""
        if self.space == "mid":
            q = latent_mean(bundle, query, max_tokens)
        else:
            q, _ = embed_block_key(bundle, query, layers="mean", max_tokens=max_tokens)
        return _norm(q.float().to(self.device) - self.gmean[self.space])

    @torch.no_grad()
    def retrieve(self, bundle, query: str, k_clusters: int = 3, k: int = 6,
                 max_tokens: int = 256) -> tuple[list[tuple[CodeNode, float, str]], list[dict]]:
        """Двухстадийный поиск: группы по центроиду → residual-листья внутри.

        Returns:
            (top-k [(node, score, "leaf")], диагностика выбранных групп
            [{"cluster": id, "size": n, "members": [имена top-3]}]).
        """
        dq = self.query_vec(bundle, query, max_tokens)
        scored_cl = sorted(((float(_norm(cl["centroid"]) @ dq), j)
                            for j, cl in enumerate(self.clusters)), reverse=True)
        picked = scored_cl[:k_clusters]
        scored: list[tuple[float, int]] = []
        for _, j in picked:
            cl = self.clusters[j]
            if len(cl["members"]) == 1:
                i = cl["members"][0]
                scored.append((float(self.dev[i] @ dq), i))
            else:
                q_res = _norm(dq - cl["centroid"])
                for i in cl["members"]:
                    scored.append((float(self.residuals[i] @ q_res), i))
        scored.sort(key=lambda t: -t[0])
        results = [(self.funcs[i], s, "leaf") for s, i in scored[:k]]
        diag = [{"cluster": j, "size": len(self.clusters[j]["members"]),
                 "members": [self.funcs[i].name for i in self.clusters[j]["members"][:3]]}
                for _, j in picked]
        return results, diag

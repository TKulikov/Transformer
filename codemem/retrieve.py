"""Unified retrieval cascade: prefetch predictions + goal tree, one ranking.

Two-level query (the risk-2 refinement): the goal embedding routes to a subtree
(coarse "what I want"), the need embedding is the centered residual query inside
(fine "what I need now"). Prefetch candidates and tree-search candidates are
merged into a SINGLE MMR-ranked list — prefetch is a candidate source, not a
privileged context section, so availability/position anchoring is removed by
construction (the risk-5 refinement).
"""

from __future__ import annotations

import torch

from .bank import CodeBank
from .prefetch import PrefetchBank
from .tree import GoalTree
from .vector import cosine_sim, mmr, normalize


class Retriever:
    """Cascade retriever over a goal tree, chunk bank and prefetch bank.

    Args:
        tree: goal hierarchy (routing + within-node search).
        bank: chunk storage (embeddings + maturity bonuses).
        prefetch: living prediction cache (step-indexed).
        top_k: default number of candidates returned.
        mmr_lambda: diversification trade-off for the unified ranking.
        centered: center keys/query by the routed node centroid.

    :meth:`retrieve` is the single entry point used by the generation loop.
    Not thread-safe.
    """

    def __init__(
        self,
        tree: GoalTree,
        bank: CodeBank,
        prefetch: PrefetchBank,
        top_k: int = 6,
        mmr_lambda: float = 0.5,
        centered: bool = True,
    ) -> None:
        self.tree = tree
        self.bank = bank
        self.prefetch = prefetch
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.centered = centered

    def retrieve(self, goal_emb: torch.Tensor, need_emb: torch.Tensor, step: int, k: int | None = None) -> list[int]:
        """Return up to k chunk ids for a plan step, unified-ranked.

        Steps: route ``goal_emb`` to a node; collect tree-search candidates
        (centered need query) and live prefetch candidates for ``step``; merge
        uniquely; re-rank ALL candidates with one MMR pass (centered need query
        + maturity bonus). Prefetch candidates carry no privilege. Empty candidate
        set gives []. ``k`` defaults to top_k.
        """
        k = self.top_k if k is None else k
        node_id = self.tree.route(goal_emb)
        node = self.tree.nodes[node_id]
        tree_ids = self.tree.search(node_id, need_emb, k)
        pre_ids = self.prefetch.scan(step)
        cand = list(dict.fromkeys(tree_ids + pre_ids))
        if not cand:
            return []
        keys = torch.stack([self.bank.get(i).emb for i in cand])
        q = need_emb.detach().float().cpu()
        if self.centered:
            q = node.centroid.center(q)
            keys = node.centroid.center(keys)
        rel = cosine_sim(q, keys) + torch.tensor([self.bank.bonus(i) for i in cand])
        sel = mmr(q, keys, k, self.mmr_lambda, relevance=rel)
        return [cand[i] for i in sel]

    def retrieve_soft(
        self,
        goal_emb: torch.Tensor,
        need_emb: torch.Tensor,
        step: int,
        k: int | None = None,
        m: int = 3,
        need_weight: float = 0.5,
        temperature: float = 0.1,
        by_centroid: bool = False,
    ) -> list[int]:
        """Soft, goal+need retrieval (attacks the goal-identification bottleneck).

        Routing signal is a blend ``normalize(normalize(goal) + need_weight *
        normalize(need))`` — the goal gives coarse context, the need points at
        which solved subgoals/functions are required now (pure need collapses,
        pure generated goal misroutes; the blend disambiguates). Soft routing
        keeps the top-m goal nodes weighted by softmax cosine; candidates are
        pooled across them and ranked by
        ``node_weight * cosine(need, chunk) + maturity_bonus`` with MMR diversity.

        Uncentered (centering is not load-bearing for code). ``need_weight=0``
        gives goal-only soft routing; larger values lean on the need. ``m`` larger
        than the goal count approaches goal-weighted flat retrieval. Returns up to
        k chunk ids; empty pool gives []. ``k`` defaults to top_k.
        """
        k = self.top_k if k is None else k
        g = normalize(goal_emb.detach().float().cpu())
        n = normalize(need_emb.detach().float().cpu())
        route_emb = normalize(g + need_weight * n)
        nodes = self.tree.route_soft(route_emb, m=m, temperature=temperature, by_centroid=by_centroid)
        weights: dict[int, float] = {}
        cand: list[int] = []
        for nid, w in nodes:
            for cid in self.tree.nodes[nid].chunk_ids:
                if cid not in weights:
                    weights[cid] = w
                    cand.append(cid)
                else:
                    weights[cid] = max(weights[cid], w)
        if not cand:
            return []
        keys = torch.stack([self.bank.get(i).emb for i in cand])
        need = need_emb.detach().float().cpu()
        rel = torch.tensor([
            weights[i] * float(cosine_sim(need, self.bank.get(i).emb[None])[0]) + self.bank.bonus(i)
            for i in cand
        ])
        sel = mmr(need, keys, k, self.mmr_lambda, relevance=rel)
        return [cand[i] for i in sel]

    def retrieve_adaptive(
        self,
        goal_emb: torch.Tensor,
        need_emb: torch.Tensor,
        step: int,
        k: int | None = None,
        m: int = 3,
        need_weight: float = 0.5,
        conf_threshold: float = 0.5,
        temperature: float = 0.1,
        by_centroid: bool = True,
    ) -> list[int]:
        """Confidence-adaptive routing: goal-routing when confident, else flat.

        Computes the routing confidence as the top cosine of the goal+need blend
        to the goal-node centroids (or labels). If confidence >= conf_threshold,
        delegates to :meth:`retrieve_soft` (goal-weighted); otherwise falls back
        to FLAT retrieval over all chunks (need query + maturity bonus + MMR).
        This guarantees flat-level performance when the goal estimate is uncertain
        (a bad goal must not hurt worse than flat) while capturing the goal-routing
        gain when confident. Returns up to k chunk ids; ``k`` defaults to top_k.
        """
        k = self.top_k if k is None else k
        g = normalize(goal_emb.detach().float().cpu())
        n = normalize(need_emb.detach().float().cpu())
        route_emb = normalize(g + need_weight * n)
        children = self.tree.nodes[self.tree.root].children if self.tree.root is not None else []

        def key_emb(nid: int) -> torch.Tensor:
            node = self.tree.nodes[nid]
            if by_centroid and node.centroid.value is not None:
                return node.centroid.value
            return node.emb

        conf = max((float(cosine_sim(route_emb, key_emb(c)[None])[0]) for c in children), default=0.0)
        if conf >= conf_threshold:
            return self.retrieve_soft(goal_emb, need_emb, step, k, m, need_weight, temperature, by_centroid)
        cand = list(dict.fromkeys(cid for nid in self.tree.nodes for cid in self.tree.nodes[nid].chunk_ids))
        if not cand:
            return []
        keys = torch.stack([self.bank.get(i).emb for i in cand])
        need = need_emb.detach().float().cpu()
        rel = cosine_sim(need, keys) + torch.tensor([self.bank.bonus(i) for i in cand])
        sel = mmr(need, keys, k, self.mmr_lambda, relevance=rel)
        return [cand[i] for i in sel]
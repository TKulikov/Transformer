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
from .vector import cosine_sim, mmr


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
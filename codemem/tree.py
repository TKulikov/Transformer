"""Goal tree: centered two-stage retrieval and goal-anchored write-back.

Nodes are goals/subgoals (depth cap 2 by default — deep trees did not help in
the dialogue semtree experiments). Each node holds a goal embedding (for coarse
routing) and a running centroid over its member code chunks (for fine, centered
within-node search). Keys are static; only centroids evolve. Write-back
assimilates a generated chunk into the matched goal when similar enough, else
creates a new solved sub-goal node (EverMemOS-style assimilate-or-create).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .bank import CodeBank
from .vector import RunningCentroid, cosine_sim, mmr


@dataclass
class GoalNode:
    """A goal/subgoal node in the tree.

    Attributes:
        id: unique node id.
        text: goal statement (or chunk text for auto-created solved nodes).
        emb: goal embedding [H], used for coarse routing.
        depth: 0 for the root goal, +1 per level.
        parent: parent node id (None for root).
        children: child node ids.
        chunk_ids: member code chunk ids stored under this goal.
        centroid: running mean of member chunk embeddings (centering key).
        solved: True for auto-created nodes representing a solved sub-goal.
    """

    id: int
    text: str
    emb: torch.Tensor
    depth: int
    parent: int | None
    children: list[int] = field(default_factory=list)
    chunk_ids: list[int] = field(default_factory=list)
    centroid: RunningCentroid = field(default_factory=RunningCentroid)
    solved: bool = False


class GoalTree:
    """Hierarchy of goals over a :class:`CodeBank`.

    Args:
        bank: chunk storage (embeddings + maturity bonuses).
        tau_assimilate: write-back similarity threshold (sim to goal >= tau
            assimilates, else a new solved node is created).
        centered: subtract the node centroid from keys/queries in search.
        mmr_lambda: diversification trade-off for within-node search.

    The first :meth:`add_goal` with parent=None establishes the root.
    :meth:`route` greedily descends by goal-embedding cosine; :meth:`search`
    does centered MMR within a node; :meth:`write_back` assimilated-or-creates.
    Not thread-safe.
    """

    def __init__(
        self,
        bank: CodeBank,
        tau_assimilate: float = 0.6,
        centered: bool = True,
        mmr_lambda: float = 0.5,
    ) -> None:
        self.bank = bank
        self.tau_assimilate = tau_assimilate
        self.centered = centered
        self.mmr_lambda = mmr_lambda
        self.nodes: dict[int, GoalNode] = {}
        self.root: int | None = None
        self._next = 0

    def add_goal(self, text: str, emb: torch.Tensor, parent: int | None = None, solved: bool = False) -> int:
        """Create a goal node; return its id. Sets the root when parent is None."""
        nid = self._next
        self._next += 1
        depth = 0 if parent is None else self.nodes[parent].depth + 1
        self.nodes[nid] = GoalNode(
            id=nid, text=text, emb=emb.detach().float().cpu().clone(),
            depth=depth, parent=parent, solved=solved,
        )
        if parent is None:
            if self.root is None:
                self.root = nid
        else:
            self.nodes[parent].children.append(nid)
        return nid

    def route(self, goal_emb: torch.Tensor) -> int:
        """Greedily descend from the root to the best-matching goal node.

        At each node, moves to the child whose goal embedding has the highest
        cosine to ``goal_emb`` iff that beats the current node's own cosine;
        otherwise stops. Returns the reached node id. Raises RuntimeError if the
        tree has no root.
        """
        if self.root is None:
            raise RuntimeError("goal tree has no root")
        node = self.root
        g = goal_emb.detach().float().cpu()
        while True:
            cur = self.nodes[node]
            if not cur.children:
                return node
            best = max(cur.children, key=lambda c: float(cosine_sim(g, self.nodes[c].emb[None])[0]))
            best_sim = float(cosine_sim(g, self.nodes[best].emb[None])[0])
            cur_sim = float(cosine_sim(g, cur.emb[None])[0])
            if best_sim > cur_sim:
                node = best
            else:
                return node

    def route_soft(self, query_emb: torch.Tensor, m: int = 3, temperature: float = 0.1) -> list[tuple[int, float]]:
        """Soft routing: top-m goal nodes weighted by softmax cosine to query.

        Unlike :meth:`route` (greedy argmax into one node), this returns several
        candidate goal nodes (the root's depth-1 children) with weights summing
        to 1, so a misestimated goal still lets the correct node contribute.
        Weights are softmax(cosine / temperature) over the top-m nodes.

        Args:
            query_emb: routing signal [H] (goal, or a goal+need blend).
            m: number of nodes to keep (clamped to the number of goal nodes, >= 1).
            temperature: softmax temperature (> 0); smaller = sharper weights.

        Returns:
            [(node_id, weight)] descending by weight. If the root has no children,
            returns [(root, 1.0)]. Raises RuntimeError if there is no root.
        """
        if self.root is None:
            raise RuntimeError("goal tree has no root")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        children = self.nodes[self.root].children
        if not children:
            return [(self.root, 1.0)]
        q = query_emb.detach().float().cpu()
        sims = [(c, float(cosine_sim(q, self.nodes[c].emb[None])[0])) for c in children]
        sims.sort(key=lambda x: -x[1])
        top = sims[: max(1, m)]
        mx = max(s for _, s in top)
        exps = [(c, math.exp((s - mx) / temperature)) for c, s in top]
        z = sum(e for _, e in exps)
        return [(c, e / z) for c, e in exps]

    def search(self, node_id: int, need_emb: torch.Tensor, k: int, lambda_: float | None = None) -> list[int]:
        """Centered MMR search within a node's chunks; return chunk ids.

        The need query and member keys are centered by the node centroid (when
        ``centered``); relevance adds the bank maturity bonus. Empty node gives
        []. ``lambda_`` defaults to the tree's mmr_lambda.
        """
        node = self.nodes[node_id]
        ids = node.chunk_ids
        if not ids:
            return []
        keys = torch.stack([self.bank.get(i).emb for i in ids])
        q = need_emb.detach().float().cpu()
        if self.centered:
            q = node.centroid.center(q)
            keys = node.centroid.center(keys)
        rel = cosine_sim(q, keys) + torch.tensor([self.bank.bonus(i) for i in ids])
        lam = self.mmr_lambda if lambda_ is None else lambda_
        sel = mmr(q, keys, k, lam, relevance=rel)
        return [ids[i] for i in sel]

    def write_back(self, cid: int, goal_emb: torch.Tensor, tau: float | None = None) -> int:
        """Assimilate a generated chunk into the tree, or create a solved node.

        Routes ``goal_emb`` to the working node, then compares the chunk
        embedding to that node's goal embedding. If sim >= tau (or the node has
        no chunks yet) the chunk joins the node and updates its centroid;
        otherwise a new solved child node is created holding the chunk. Returns
        the node id the chunk landed in. Sets the chunk's goal_id.
        """
        tau = self.tau_assimilate if tau is None else tau
        node_id = self.route(goal_emb)
        node = self.nodes[node_id]
        chunk = self.bank.get(cid)
        sim = float(cosine_sim(chunk.emb, node.emb[None])[0])
        if sim >= tau or not node.chunk_ids:
            node.chunk_ids.append(cid)
            node.centroid.update(chunk.emb)
            chunk.goal_id = node_id
            return node_id
        new_id = self.add_goal(text=chunk.text, emb=chunk.emb, parent=node_id, solved=True)
        new = self.nodes[new_id]
        new.chunk_ids.append(cid)
        new.centroid.update(chunk.emb)
        chunk.goal_id = new_id
        return new_id
"""Goal acquisition: generate, compress, embed, and decompose a coding goal.

Pipeline (``для кода.md`` + risk refinements): the request -> a generated goal
statement -> a compressed goal (models are verbose; compression is measured by
embedding fidelity cos(full, compressed)) -> a shallow goal tree via recursive
binary decomposition. A split is accepted only when the dendrogram root gap
exceeds tau_split AND a self-sufficiency judge confirms both subgoals are
independently implementable (risk-3: a binary, well-posed question instead of an
open-ended atomicity judgement). Depth is capped (default 2).

The recursive decomposition logic (:func:`decompose`) is pure and injectable
(judge callback), so it is unit-testable offline without the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .cluster import dendrogram, root_split
from .config import Config
from .model import CodeModel
from .vector import cosine_sim

# judge(texts_a, texts_b) -> True iff both subgoal groups are self-sufficient
Judge = Callable[[list[str], list[str]], bool]


@dataclass
class Goal:
    """A coding goal derived from a request.

    Attributes:
        text: full generated goal statement.
        compressed: shortened goal used for embedding/routing.
        emb: embedding of the compressed goal [H].
        fidelity: cos(embed(full), embed(compressed)); low values flag lossy
            compression (risk-1 metric).
    """

    text: str
    compressed: str
    emb: torch.Tensor
    fidelity: float


@dataclass
class SubGoal:
    """A node in the decomposed goal tree (children of the root goal).

    Attributes:
        text: subgoal statement (joined member texts for internal nodes).
        emb: subgoal embedding [H] (mean of members for internal nodes).
        children: nested subgoals (empty for atomic leaves).
        atomic: True if this is an indivisible unit (no accepted split).
    """

    text: str
    emb: torch.Tensor
    children: list["SubGoal"] = field(default_factory=list)
    atomic: bool = True


def decompose(
    items: list[tuple[str, torch.Tensor]],
    tau_split: float,
    judge: Judge,
    depth_cap: int,
    depth: int = 0,
) -> list[SubGoal]:
    """Recursively split candidate subgoals into a shallow binary tree (pure).

    Args:
        items: (text, embedding [H]) pairs at the current level.
        tau_split: minimum dendrogram root-gap to consider a split.
        judge: callback deciding whether both groups of a split are
            self-sufficient; a split is accepted only if it returns True.
        depth_cap: maximum recursion depth (0 = everything atomic).
        depth: current depth (internal recursion parameter).

    Returns:
        Top-level SubGoal nodes. With <= 1 item, depth >= depth_cap, no clean
        gap, or a rejected judge, returns atomic leaves (one per item). Otherwise
        returns two internal nodes, each recursively decomposed. Deterministic.
    """
    if len(items) <= 1 or depth >= depth_cap:
        return [SubGoal(text=t, emb=e.clone(), atomic=True) for t, e in items]
    embs = torch.stack([e for _, e in items])
    split = root_split(dendrogram(embs), len(items))
    if split is None:
        return [SubGoal(text=t, emb=e.clone(), atomic=True) for t, e in items]
    ga, gb, gap = split
    if gap < tau_split:
        return [SubGoal(text=t, emb=e.clone(), atomic=True) for t, e in items]
    texts_a = [items[i][0] for i in ga]
    texts_b = [items[i][0] for i in gb]
    if not judge(texts_a, texts_b):
        return [SubGoal(text=t, emb=e.clone(), atomic=True) for t, e in items]
    node_a = SubGoal(
        text="; ".join(texts_a), emb=embs[list(ga)].mean(0),
        children=decompose([items[i] for i in ga], tau_split, judge, depth_cap, depth + 1),
        atomic=False,
    )
    node_b = SubGoal(
        text="; ".join(texts_b), emb=embs[list(gb)].mean(0),
        children=decompose([items[i] for i in gb], tau_split, judge, depth_cap, depth + 1),
        atomic=False,
    )
    return [node_a, node_b]


class GoalBuilder:
    """Model-backed goal acquisition and decomposition.

    Args:
        model: loaded :class:`CodeModel`.
        cfg: hyperparameters (token budgets, tau_split, depth_cap, embed layer).

    :meth:`build` produces a :class:`Goal`; :meth:`subgoals` proposes candidate
    subgoals and decomposes them with a model self-sufficiency judge.
    """

    def __init__(self, model: CodeModel, cfg: Config) -> None:
        self.model = model
        self.cfg = cfg

    def build(self, request: str) -> Goal:
        """Generate a goal from the request, compress it, and measure fidelity."""
        full = self.model.generate(
            "State the concrete engineering goal of this request in one or two "
            f"sentences. Request: {request}",
            max_new_tokens=self.cfg.goal_max_tokens,
        )
        compressed = self.model.generate(
            "Compress this goal to a single short phrase (<= 12 words), keeping "
            f"the key technical intent. Goal: {full}",
            max_new_tokens=self.cfg.compress_max_tokens,
        )
        full_emb, comp_emb = self.model.embed([full, compressed])
        fidelity = float(cosine_sim(full_emb, comp_emb[None])[0])
        return Goal(text=full, compressed=compressed, emb=comp_emb, fidelity=fidelity)

    def _judge(self, texts_a: list[str], texts_b: list[str]) -> bool:
        """Ask the model whether both subgoal groups are self-sufficient."""
        a = "; ".join(texts_a)
        b = "; ".join(texts_b)
        out = self.model.generate(
            "Two subgoals:\nA: " + a + "\nB: " + b +
            "\nIs EACH subgoal self-contained enough to implement independently? "
            "Answer only YES or NO.",
            max_new_tokens=8,
        )
        return "yes" in out.lower()

    def subgoals(self, goal: Goal) -> list[SubGoal]:
        """Propose candidate subgoals for a goal and decompose them into a tree."""
        out = self.model.generate(
            "List 3-6 concrete subgoals to achieve this goal, one per line, "
            f"numbered. Goal: {goal.compressed}",
            max_new_tokens=self.cfg.goal_max_tokens,
        )
        texts = []
        for line in out.splitlines():
            line = line.strip()
            if line and line[0].isdigit():
                texts.append(line.lstrip("0123456789.)- ").strip())
        if len(texts) < 2:
            return []
        embs = self.model.embed(texts)
        items = list(zip(texts, embs))
        return decompose(items, self.cfg.tau_split, self._judge, self.cfg.depth_cap)
"""Chunked code generation driven by plan steps and goal-conditioned memory.

The generation loop (``для кода.md``): build a goal tree from the request, then
for each code chunk generate a short plan ("what I want + what I need"), predict
candidate functions into the prefetch bank, retrieve a unified candidate list,
inject it as text context, generate the chunk, and write it back into the goal
tree by similarity. Retrieved candidates accrue use-based maturity scores.

Placeholder strategies (documented, to refine): prediction defaults to a pure
tree search (model-based prediction is a later refinement); use-success defaults
to "chunk generated" (real success = tests pass, wired in the eval phase).
"""

from __future__ import annotations

import torch

from .bank import CodeBank
from .config import Config
from .goal import Goal, GoalBuilder, SubGoal
from .model import CodeModel
from .prefetch import PrefetchBank
from .retrieve import Retriever
from .tree import GoalTree


class CodePlanner:
    """Orchestrates goal-conditioned chunked code generation.

    Args:
        model: loaded generator/embedder.
        cfg: hyperparameters.
        bank, tree, prefetch, retriever: the memory components (tree must share
            ``bank``; retriever must share tree/bank/prefetch).
        goal_builder: goal acquisition (defaults to GoalBuilder(model, cfg)).

    :meth:`run` is the end-to-end entry point; :meth:`generate_chunk` produces one
    chunk with full memory wiring. Not thread-safe.
    """

    def __init__(
        self,
        model: CodeModel,
        cfg: Config,
        bank: CodeBank,
        tree: GoalTree,
        prefetch: PrefetchBank,
        retriever: Retriever,
        goal_builder: GoalBuilder | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.bank = bank
        self.tree = tree
        self.prefetch = prefetch
        self.retriever = retriever
        self.goal_builder = goal_builder or GoalBuilder(model, cfg)

    def _materialize(self, nodes: list[SubGoal], parent: int) -> None:
        """Add decomposed subgoals to the goal tree under ``parent`` (recursive)."""
        for n in nodes:
            nid = self.tree.add_goal(n.text, n.emb, parent=parent)
            if n.children:
                self._materialize(n.children, nid)

    def plan_step(self, goal: Goal, written: str, step: int) -> tuple[str, str]:
        """Generate a pre-chunk plan; return (want, need).

        Prompts the model for a WANT/NEED pair given the goal and code so far.
        Falls back to the raw plan text as ``need`` when parsing finds no NEED
        line, so retrieval always has a query.
        """
        out = self.model.generate(
            f"Goal: {goal.compressed}\nCode so far:\n{written[-1500:]}\n"
            "Before the next code chunk, state what you want to achieve and what "
            "you need from existing code. Format exactly:\nWANT: ...\nNEED: ...",
            max_new_tokens=self.cfg.plan_max_tokens,
        )
        want, need = out, ""
        for line in out.splitlines():
            u = line.upper()
            if u.startswith("WANT:"):
                want = line[5:].strip()
            elif u.startswith("NEED:"):
                need = line[5:].strip()
        return want, (need or out)

    def predict(self, goal_emb: torch.Tensor, need_emb: torch.Tensor, step: int) -> list[int]:
        """Predict candidate chunk ids for a step (prefetch population).

        Default strategy: pure tree search (no prefetch) using the need query —
        a placeholder for model-based prediction. Returns up to prefetch_size ids.
        """
        node = self.tree.route(goal_emb)
        return self.tree.search(node, need_emb, self.cfg.prefetch_size)

    def generate_chunk(self, goal: Goal, written: str, step: int) -> str:
        """Generate one code chunk with full memory wiring; return its text.

        Plan -> need embedding -> predict+prefetch -> unified retrieval -> text
        context injection -> generation -> embed -> bank.add -> tree.write_back
        -> record_use for retrieved candidates -> prefetch.advance. Side effects:
        mutates bank (new chunk + use scores), tree (write-back), prefetch.
        """
        _, need = self.plan_step(goal, written, step)
        need_emb = self.model.embed([need])[0]
        self.prefetch.put(step, self.predict(goal.emb, need_emb, step))
        cand = self.retriever.retrieve(goal.emb, need_emb, step)
        context = "\n\n".join(self.bank.get(i).text for i in cand)
        prompt = (
            f"Goal: {goal.compressed}\nAlready written:\n{written[-1500:]}\n"
            f"Next step need: {need}\nReusable code from memory:\n{context}\n"
            "Write the next code chunk fulfilling the need. Output only code."
        )
        chunk = self.model.generate(prompt, max_new_tokens=self.cfg.chunk_max_tokens)
        chunk_emb = self.model.embed([chunk])[0]
        cid = self.bank.add(chunk, chunk_emb)
        self.tree.write_back(cid, goal.emb)
        for i in cand:
            self.bank.record_use(i, success=bool(chunk.strip()))
        self.prefetch.advance(step + 1)
        return chunk

    def run(self, request: str, n_chunks: int = 3) -> tuple[Goal, list[str]]:
        """End-to-end: build goal tree from a request, generate n_chunks chunks.

        Returns (goal, chunks). Builds the root goal, decomposes and materializes
        subgoals, then generates chunks sequentially (each sees prior code).
        """
        goal = self.goal_builder.build(request)
        root = self.tree.add_goal(goal.compressed, goal.emb)
        self._materialize(self.goal_builder.subgoals(goal), root)
        written = ""
        chunks: list[str] = []
        for step in range(n_chunks):
            chunk = self.generate_chunk(goal, written, step)
            chunks.append(chunk)
            written += "\n" + chunk
        return goal, chunks
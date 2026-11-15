"""Static code-chunk storage with maturity tiers and use-based scoring.

The code branch stores a codebase incrementally: chunks (functions/methods) are
valuable even when not independently executable (a method of a half-written
class cannot be run), so there is NO execution gate at write time. Instead a
chunk matures WRITTEN -> USED -> VERIFIED (retroactive promotion when tests
pass) and earns a retrieval bonus from successful uses. Embeddings are static —
only the scores move (static storage + dynamic selection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import torch


class Tier(IntEnum):
    """Maturity of a stored chunk. Higher = more trusted at retrieval."""

    WRITTEN = 0     # generated, not yet used or tested
    USED = 1        # retrieved and applied at least once
    VERIFIED = 2    # passed tests / execution once complete


@dataclass
class Chunk:
    """One stored code chunk.

    Attributes:
        id: unique integer id assigned by :class:`CodeBank`.
        text: source code of the chunk.
        emb: static embedding [H]; never mutated after insertion.
        goal_id: id of the goal/problem this chunk serves (None if unassigned).
        tier: current maturity tier.
        uses: number of times retrieved and applied.
        successes: number of those uses that succeeded (tests passed / accepted).
    """

    id: int
    text: str
    emb: torch.Tensor
    goal_id: int | None = None
    tier: Tier = Tier.WRITTEN
    uses: int = 0
    successes: int = 0


class CodeBank:
    """Store of code chunks with static keys and dynamic maturity scores.

    Args:
        use_boost: retrieval score bonus per successful use.
        verified_boost: extra bonus for VERIFIED chunks.

    Lifecycle: :meth:`add` inserts a chunk; :meth:`record_use` logs an
    application (auto-promotes WRITTEN -> USED on first use); :meth:`promote`
    sets a tier explicitly (e.g. VERIFIED after tests). :meth:`bonus` is the
    additive retrieval score. Not thread-safe.
    """

    def __init__(self, use_boost: float = 0.05, verified_boost: float = 0.1) -> None:
        self.use_boost = use_boost
        self.verified_boost = verified_boost
        self._chunks: dict[int, Chunk] = {}
        self._next = 0

    def add(self, text: str, emb: torch.Tensor, goal_id: int | None = None) -> int:
        """Insert a chunk with a static embedding [H]; return its new id.

        The embedding is detached/cloned to CPU float32 so later bank operations
        never mutate the caller's tensor or the stored key.
        """
        cid = self._next
        self._next += 1
        self._chunks[cid] = Chunk(
            id=cid, text=text, emb=emb.detach().float().cpu().clone(), goal_id=goal_id,
        )
        return cid

    def get(self, cid: int) -> Chunk:
        """Return the chunk with ``cid``. Raises KeyError if absent."""
        return self._chunks[cid]

    def record_use(self, cid: int, success: bool) -> None:
        """Log one application of a chunk.

        Increments ``uses`` (and ``successes`` when ``success``). First use
        promotes WRITTEN -> USED; VERIFIED is never downgraded. Raises KeyError
        if ``cid`` is absent.
        """
        ch = self._chunks[cid]
        ch.uses += 1
        if success:
            ch.successes += 1
        if ch.tier == Tier.WRITTEN:
            ch.tier = Tier.USED

    def promote(self, cid: int, tier: Tier) -> None:
        """Set a chunk's maturity tier (e.g. VERIFIED after tests pass).

        Only promotes (never downgrades): a lower tier than current is ignored.
        Raises KeyError if ``cid`` is absent.
        """
        ch = self._chunks[cid]
        if tier > ch.tier:
            ch.tier = tier

    def bonus(self, cid: int) -> float:
        """Additive retrieval score: use_boost*successes + verified_boost if VERIFIED."""
        ch = self._chunks[cid]
        b = self.use_boost * ch.successes
        if ch.tier == Tier.VERIFIED:
            b += self.verified_boost
        return b

    def __len__(self) -> int:
        return len(self._chunks)
"""Living prediction bank: plan-step prefetch of candidate functions.

The code branch's "living memory" is NOT mutating vectors (those drifted and
destroyed retrievability in the dialogue experiments). It is short-lived
*predictions* of what a plan step will need: before generating a chunk, candidate
functions are prefetched here based on the step's plan. Predictions live for a
fixed number of plan steps (TTL tied to steps, not wall time) and are scanned as
one candidate source among others — never a privileged context section.
"""

from __future__ import annotations


class PrefetchBank:
    """Step-indexed cache of predicted candidate chunk ids.

    Args:
        ttl_steps: how many plan steps a prediction batch stays live (>= 1).
            A batch put at step s is scannable at steps s .. s+ttl_steps-1.

    Lifecycle: :meth:`put` records predictions for a step; :meth:`scan` returns
    live ids for the current step; :meth:`advance` evicts expired batches.
    Keys (chunk ids) are never mutated. Not thread-safe.
    """

    def __init__(self, ttl_steps: int = 1) -> None:
        if ttl_steps < 1:
            raise ValueError("ttl_steps must be >= 1")
        self.ttl_steps = ttl_steps
        self._batches: dict[int, list[int]] = {}

    def put(self, step: int, chunk_ids: list[int]) -> None:
        """Record predicted candidate ids for ``step`` (overwrites that step)."""
        self._batches[step] = list(chunk_ids)

    def scan(self, step: int) -> list[int]:
        """Return live candidate ids for ``step``, deduplicated, order-stable.

        Includes batches put at steps s with step-ttl_steps < s <= step. Expired
        batches are ignored (but not removed until :meth:`advance`).
        """
        lo = step - self.ttl_steps
        seen: dict[int, None] = {}
        for s in sorted(self._batches):
            if lo < s <= step:
                for cid in self._batches[s]:
                    seen.setdefault(cid, None)
        return list(seen)

    def advance(self, step: int) -> None:
        """Evict batches that can no longer be scanned at ``step`` or later."""
        lo = step - self.ttl_steps
        for s in [s for s in self._batches if s <= lo]:
            del self._batches[s]

    def __len__(self) -> int:
        return len(self._batches)
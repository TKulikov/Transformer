"""codemem — goal-conditioned memory for code generation (code branch).

A standalone system, separate from the dialogue ``tmem`` package. Shared idea
only: centering + static storage + dynamic selection. Architecture follows
``для кода.md`` and the risk refinements agreed on 15.11:

* goal from request -> compression -> shallow goal tree (centered keys,
  dendrogram + self-sufficiency judge, depth cap 2);
* per-chunk plan ("what I want + what I need") drives a retrieval cascade:
  prefetch predictions + goal tree, merged into ONE MMR-ranked candidate list;
* write-back by goal similarity; living components are prefetch predictions
  and use-based maturity scores — keys never mutate.

See ``docs/PLAN-code-branch.md`` for the phased build plan.
"""

from __future__ import annotations

from .config import Config

__all__ = ["Config"]
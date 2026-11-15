"""Hyperparameters for codemem, gathered so experiments can sweep them.

Defaults are grounded in the dialogue-branch findings: a mid full-attention
embedding layer, MMR diversification (lambda=0.5), and a shallow goal tree
(depth cap 2 — deep trees did not help in the semtree experiments).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """All tunable knobs for the code-branch memory system.

    Attributes:
        repo: HuggingFace model id (loaded offline from the local cache).
        embed_layer: hidden-state layer used for embeddings. Qwen3.5-4B has 32
            layers with full attention at 3/7/11/15/19/23/27/31; 15 is a mid
            full-attention layer. Index follows ``output_hidden_states`` (0 is
            the embedding output, so layer L means hidden_states[L]).
        pooling: token pooling for embeddings, "mean" (mask-aware) or "last".
        dtype: model dtype string ("bfloat16" recommended on ROCm).
        device: torch device string ("cuda" for the ROCm card, "cpu" for tests).
        centered: subtract the node running centroid from keys and queries
            (deviation space). The load-bearing mechanism from the dialogue work.
        mmr_lambda: MMR trade-off in [0, 1]; 1 = pure relevance, 0 = pure
            diversity. 0.5 was the robust never-worse setting against clone flood.
        top_k: number of candidates returned by retrieval.
        depth_cap: maximum goal-tree depth (root goal = depth 0). 2 by default.
        tau_split: minimum dendrogram merge-height gap required before asking the
            self-sufficiency judge whether a goal splits. Filters spurious splits.
        tau_assimilate: write-back similarity threshold; sim >= tau assimilates a
            chunk into the matched node, otherwise a new node/goal is created.
        prefetch_size: max candidate functions predicted per plan step.
        prefetch_ttl_steps: how many plan steps a prediction batch stays live.
        use_boost: retrieval score bonus per recorded successful use of a chunk.
        verified_boost: extra score bonus for chunks in the VERIFIED maturity tier.
        chunk_max_tokens: generation budget per code chunk.
        plan_max_tokens: generation budget for a pre-chunk plan step.
        goal_max_tokens: generation budget for the initial goal statement.
        compress_max_tokens: generation budget for the compressed goal.
    """

    repo: str = "Qwen/Qwen3.5-4B"
    embed_layer: int = 15
    pooling: str = "mean"
    dtype: str = "bfloat16"
    device: str = "cuda"

    centered: bool = True
    mmr_lambda: float = 0.5
    top_k: int = 6

    depth_cap: int = 2
    tau_split: float = 0.15
    tau_assimilate: float = 0.6

    prefetch_size: int = 6
    prefetch_ttl_steps: int = 1

    use_boost: float = 0.05
    verified_boost: float = 0.1

    chunk_max_tokens: int = 256
    plan_max_tokens: int = 96
    goal_max_tokens: int = 128
    compress_max_tokens: int = 48
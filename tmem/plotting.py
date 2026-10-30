"""Matplotlib reports built from *measured* run data (no simulated curves,
unlike the notebook's final plot cells)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .dialogue import DialogueRun  # noqa: E402


def plot_context_growth(runs: dict[str, DialogueRun], out_path: str) -> str:
    """Prompt tokens per turn per strategy."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, run in runs.items():
        tokens = [t.prompt_tokens for t in run.turns]
        ax.plot(range(1, len(tokens) + 1), tokens, marker="o", label=name)
    ax.set_xlabel("Dialogue turn")
    ax.set_ylabel("Prompt tokens")
    ax.set_title("Per-turn prompt size by strategy (measured)")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_cache_growth(runs: dict[str, DialogueRun], out_path: str) -> str:
    """KV-cache length per decode step, longest turn."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, run in runs.items():
        longest = max((t.decode.cache_lengths for t in run.turns), key=len, default=[])
        ax.plot(range(len(longest)), longest, marker="s", label=name)
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Cached slots")
    ax.set_title("KV-cache growth during the longest turn (measured)")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_efficiency_quality(runs: dict[str, DialogueRun], out_path: str) -> str:
    """Token cost vs probe quality bars."""
    names = list(runs)
    tokens = [runs[n].total_prompt_tokens for n in names]
    probes = [runs[n].summary()["probe_mean"] or 0.0 for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(names, tokens, color="tab:red", alpha=0.8)
    ax1.set_title("Total prompt tokens (lower = cheaper)")
    ax1.tick_params(axis="x", rotation=30)
    ax2.bar(names, probes, color="tab:blue", alpha=0.8)
    ax2.set_title("Mean probe score (higher = answers kept)")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

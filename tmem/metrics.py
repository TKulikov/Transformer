"""Quality and efficiency metrics.

Answer quality follows the heuristics used in the original notebook
(lexical novelty vs dialogue history, strict variant with an internal
diversity penalty), plus a keyword scorer used to compare generated
answers against pre-scripted expected answers.
"""

from __future__ import annotations

from statistics import mean


def keyword_score(text: str, expected: list[str]) -> float:
    """Fraction of expected keywords found in ``text`` (case-insensitive)."""
    if not expected:
        return 0.0
    low = text.lower()
    hit = sum(1 for kw in expected if kw.lower() in low)
    return round(hit / len(expected), 4)


def novelty_quality(text: str, history: list[str]) -> float:
    """Notebook metric for T5/GPT-2 runs: length bonus + word novelty."""
    if len(text) < 10:
        return 0.1
    words = set(text.lower().split())
    history_words = set(" ".join(history).lower().split())
    novelty = len(words - history_words) / max(len(words), 1)
    return round(min(1.0, (len(text) / 200) + novelty), 2)


def strict_quality(text: str, history: list[str]) -> float:
    """Notebook's "strict" metric for Qwen: length + diversity + novelty."""
    if len(text) < 20:
        return 0.05
    words = text.lower().split()
    if not words:
        return 0.05
    unique = set(words)
    internal_diversity = len(unique) / len(words)
    history_words = set(" ".join(history).lower().split())
    novelty = len(unique - history_words) / len(unique)
    score = (len(text) / 400) * 0.3 + internal_diversity * 0.4 + novelty * 0.3
    return round(min(1.0, score), 2)


def summarize(values: list[float]) -> dict:
    """mean/min/max over a metric series (safe on empty)."""
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {"mean": round(float(mean(values)), 4), "min": round(min(values), 4), "max": round(max(values), 4)}

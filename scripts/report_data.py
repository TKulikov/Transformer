"""Load the single comparable Qwen3.5 series used by all final documents."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "report"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
DATA = RES / "qwen35_comparison" / "qwen35-4b_battery100.json"

ORDER = [
    "classic_kv_full",
    "sparse_sliding_128",
    "sparse_topk_128",
    "snapkv_128",
    "pyramidkv_128",
    "rag_text_only",
    "v4_rag_kv",
]

LABELS = {
    "classic_kv_full": "Чистый Qwen: полный контекст + KV",
    "sparse_sliding_128": "StreamingLLM",
    "sparse_topk_128": "H2O",
    "snapkv_128": "SnapKV",
    "pyramidkv_128": "PyramidKV",
    "rag_text_only": "Текстовый RAG-контроль",
    "v4_rag_kv": "Двухканальная память",
}


def load(path: Path = DATA) -> dict:
    return json.loads(path.read_text())


def q35_all() -> dict:
    data = load()
    session = next(iter(data["sessions"].values()))
    return {name: session["strategies"][name] for name in ORDER if name in session["strategies"]}


def fmt(value, spec="{:.3f}", dash="—") -> str:
    return dash if value is None else spec.format(value)


def quality_rows(q: dict) -> list[list[str]]:
    rows = []
    for key in ORDER:
        d = q.get(key)
        if d:
            rows.append([
                LABELS[key], fmt(d["recall_mean"]), fmt(d.get("recall_direct")),
                fmt(d.get("recall_paraphrase")), fmt(d.get("tone_mean")),
                fmt(d.get("stale_rate")), fmt(d.get("rep_mean")),
            ])
    return rows


def cost_rows(q: dict) -> list[list[str]]:
    reference = q["classic_kv_full"]["cache_slots"]
    rows = []
    for key in ORDER:
        d = q.get(key)
        if d:
            factor = reference / d["cache_slots"] if d["cache_slots"] else None
            rows.append([
                LABELS[key], f"{d['prompt_tokens']:,}", f"{d['generated_tokens']:,}",
                f"{d.get('embed_tokens', 0):,}", f"{d['cache_slots']:,}", fmt(factor, "{:.2f}×"),
                fmt(d["mean_turn_latency_s"], "{:.2f}"),
                fmt(d.get("median_turn_latency_s"), "{:.2f}"),
                fmt(d.get("p90_turn_latency_s"), "{:.2f}"),
                str(d["max_cache_len"]), fmt(d.get("peak_mem_mb"), "{:.0f}"),
            ])
    return rows


QUALITY_HEADERS = ["Система", "Recall", "Прямые", "Перефраз.", "Тон", "Stale", "Повторы"]
COST_HEADERS = [
    "Система", "Prompt", "Генер.", "Embed", "Attn-слоты", "Экономия", "Mean, с", "Median, с",
    "P90, с", "Max KV", "Peak MiB",
]

"""Load the historical Qwen3.5 battery and current corpus series for final documents."""

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
    "v4_kv_mid": "Живая память k32",
    "v4_kv_mid_dyn": "Живая память dynamic",
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


CORE_DIR = RES / "corpus" / "full-core-v1"
RAG_DIR = RES / "corpus" / "full-rag-v1"
CORE_ORDER = [
    "classic_kv_full", "sparse_sliding_128", "sparse_topk_128", "snapkv_128",
    "pyramidkv_128", "v4_kv_mid", "v4_kv_mid_dyn",
]
RAG_PURE_ORDER = [
    "classic_kv_full", "rag_text_only", "rag_last_centered", "rag_latent_mid",
    "rag_mid_centered", "rag_mean", "rag_mean_centered", "rag_bank_key", "rag_bank_dev",
]
RAG_PAIRS = [
    ("last raw", "rag_text_only", "v4_rag_kv_plain"),
    ("last centered", "rag_last_centered", "v4_rag_kv"),
    ("mid raw", "rag_latent_mid", "v4_rag_kv_mid_plain"),
    ("mid centered", "rag_mid_centered", "v4_rag_kv_mid"),
    ("mean raw", "rag_mean", "v4_rag_kv_mean"),
    ("mean centered", "rag_mean_centered", "v4_rag_kv_mean_centered"),
]
RAG_CENTERING_PAIRS = [
    ("rag_last_centered", "rag_text_only"),
    ("rag_mid_centered", "rag_latent_mid"),
    ("rag_mean_centered", "rag_mean"),
]


def corpus_data(run_dir: Path, order: list[str]) -> dict:
    """Load completed corpus batches and compute article-ready strategy metrics."""
    from statistics import mean, median

    from tmem.corpus_stats import load_run, summarize

    run = load_run(run_dir)
    records = sorted(run["records"], key=lambda rec: rec["batch_id"])
    summary = summarize(run)
    metrics = {}
    for name in order:
        entries = [rec["strategies"][name] for rec in records]
        analyses = [entry["analysis"] for entry in entries]
        answerable = []
        for analysis in analyses:
            values = [probe["score"] or 0.0 for probe in analysis["probe_details"]
                      if probe["kind"] != "unanswerable"]
            answerable.append(mean(values) if values else 0.0)
        item = summary["strategies"][name]
        item.update({
            "answerable": mean(answerable),
            "peak": mean(analysis["peak_mem_mb"] for analysis in analyses),
            "kv_injected": sum(analysis.get("kv_injected_slots") or 0 for analysis in analyses),
            "kv_steps": sum(analysis.get("kv_steps") or 0 for analysis in analyses),
        })
        item["slots_per_step"] = item["kv_injected"] / item["kv_steps"] if item["kv_steps"] else None
        metrics[name] = item
    return {"run": run, "records": records, "metrics": metrics, "summary": summary}


def core_data() -> dict:
    """Return current core metrics and Holm-corrected dynamic-memory comparisons."""
    from tmem.corpus_stats import compare_pairs

    data = corpus_data(CORE_DIR, CORE_ORDER)
    pairs = [("v4_kv_mid_dyn", "v4_kv_mid"), ("v4_kv_mid_dyn", "pyramidkv_128"),
             ("v4_kv_mid_dyn", "snapkv_128")]
    data["pvalues"] = compare_pairs(data["run"], pairs)
    return data


def rag_data() -> dict:
    """Return current RAG metrics and paired centering/hybrid comparisons."""
    from tmem.corpus_stats import compare_pairs

    hybrid_names = [dual for _, _, dual in RAG_PAIRS]
    data = corpus_data(RAG_DIR, RAG_PURE_ORDER + hybrid_names)
    data["centering_pvalues"] = compare_pairs(data["run"], RAG_CENTERING_PAIRS)
    data["hybrid_pvalues"] = compare_pairs(data["run"], [(dual, pure) for _, pure, dual in RAG_PAIRS])
    pair_rows = []
    for label, pure, dual in RAG_PAIRS:
        deltas = []
        cost_deltas = []
        for rec in data["records"]:
            pure_entry = rec["strategies"][pure]
            dual_entry = rec["strategies"][dual]
            deltas.append(dual_entry["analysis"]["recall_mean"] - pure_entry["analysis"]["recall_mean"])
            cost_deltas.append((dual_entry["e2e"]["median_s"] / pure_entry["e2e"]["median_s"] - 1) * 100)
        from statistics import mean, median
        pair_rows.append({
            "label": label, "pure": pure, "dual": dual,
            "pure_recall": data["metrics"][pure]["recall_mean"],
            "dual_recall": data["metrics"][dual]["recall_mean"],
            "delta": mean(deltas), "deltas": deltas,
            "e2e_delta": median(cost_deltas),
            "pvalue": data["hybrid_pvalues"][f"{dual} vs {pure}"],
        })
    data["pair_rows"] = pair_rows
    return data


def core_table(data: dict) -> list[list[str]]:
    """Format core strategy rows as recall, answerable recall, E2E, VRAM and prompt count."""
    rows = []
    for name in CORE_ORDER:
        item = data["metrics"][name]
        lo, hi = item["recall_ci95"]
        rows.append([LABELS.get(name, name), f"{item['recall_mean']:.3f} [{lo:.3f}; {hi:.3f}]",
                     f"{item['answerable']:.3f}", f"{item['e2e_latency_median_s']:.3f}",
                     f"{item['peak']:.0f}", f"{item['prompt_tokens']:.0f}"])
    return rows


def rag_pure_table(data: dict) -> list[list[str]]:
    """Format RAG-only strategy rows without including K/V hybrids."""
    rows = []
    for name in RAG_PURE_ORDER:
        item = data["metrics"][name]
        lo, hi = item["recall_ci95"]
        rows.append([name if name != "classic_kv_full" else "Full KV",
                     f"{item['recall_mean']:.3f} [{lo:.3f}; {hi:.3f}]", f"{item['answerable']:.3f}",
                     f"{item['e2e_latency_median_s']:.3f}", f"{item['peak']:.0f}", f"{item['prompt_tokens']:.0f}"])
    return rows


def rag_hybrid_table(data: dict) -> list[list[str]]:
    """Format paired RAG/K/V effects with dialogue-level Holm-adjusted p-values."""
    return [[row["label"], f"{row['pure_recall']:.3f}", f"{row['dual_recall']:.3f}",
             f"{row['delta']:+.3f}", f"{row['pvalue']:.4f}", f"{row['e2e_delta']:+.1f}%"]
            for row in data["pair_rows"]]

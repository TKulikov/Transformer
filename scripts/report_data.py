"""Build the full project report: figures, tables, PDF (+ markdown twin).

Sources: results/stress*, results/battery*, results/battery/trainreader_run*.log
Run:  python scripts/build_report.py
Out:  results/report/Memory_System_Report.pdf / .md + figs/*.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "report"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- labels & data
LABELS = {
    "no_cache_full": "Пересчёт без кэша",
    "classic_kv_full": "Эталон: полный контекст + классический KV-кэш",
    "sparse_sliding_128": "Скользящий KV-кэш (128 слотов)",
    "sparse_topk_128": "Разреженный KV-кэш top-k/H2O (128)",
    "sparse_sliding_32": "Скользящий KV-кэш (32 слота)",
    "window2_only": "Короткий буфер без памяти (2 хода)",
    "window5_only": "Буфер без памяти (5 ходов)",
    "v1_memory": "Ступень 1 (ноутбук): скрытые темы + аддитивный хук",
    "window2_mem_def": "Ступень 1 (ноутбук): скрытые темы + аддитивный хук",
    "v2_memory_ema": "Ступень 1': хук с EMA-слиянием (k=.25, порог .95)",
    "window2_mem_beta50": "Ступень 1: хук, сила .5",
    "window2_mem_small16": "Ступень 1: хук, банк 16 слотов",
    "window2_mem_retrieve16": "Ступень 1: хук, выдавать 16 слотов",
    "window2_mem_crit10": "Ступень 1: хук, строгий порог гейта",
    "v2_rag_plain": "Ступень 2: банк фактов текстом в промпт (RAG)",
    "v2_rag_gist": "Ступень 2: банк + темы из ступени 1 для выдачи",
    "v2_full": "Ступень 2: банк+темы, буфер 5",
    "v3_rag_centroid": "Ступень 3: банк с центрированием по теме",
    "v3_centroid_plus_gist": "Ступень 3: центроид + темы из ступени 1",
    "v3w5_rag_centroid": "Ступень 3: центроид, буфер 5",
    "v4_kv_mid": "Ступень 4: только память в attention (без банка)",
    "v4_rag_kv": "Ступень 4: банк + память в attention (финальная)",
}

SHORT = {
    "classic_kv_full": "Эталон (полный контекст)",
    "window2_only": "Буфер без памяти",
    "v1_memory": "1: хук активаций",
    "window2_mem_def": "1: хук активаций",
    "v2_memory_ema": "1': EMA-хук",
    "v2_rag_plain": "2: банк текстом",
    "v2_rag_gist": "2: банк+gist",
    "v2_full": "2: полный v2",
    "v3_rag_centroid": "3: центроид",
    "v3_centroid_plus_gist": "3: центроид+gist",
    "v3w5_rag_centroid": "3: центроид буфер5",
    "v4_kv_mid": "4: память в attention",
    "v4_rag_kv": "4: банк+attention",
}

def load(path):
    try:
        return json.load((RES / path).open())
    except Exception:
        return None


def grab(data, session_key_contains, strategy):
    """strategy metrics dict from a battery json, tolerant to session names."""
    if not data:
        return None
    for sname, sess in data.get("sessions", {}).items():
        if session_key_contains in sname and strategy in sess["strategies"]:
            return sess["strategies"][strategy]
    return None


# qwen3.5 final table: prefer the vectorized re-run for shared strategies
Q35 = {}
v4vec = load("battery/qwen35-4b_battery100_v4_vec.json")
v3 = load("battery/qwen35-4b_battery100_v3.json")
abl = load("battery/qwen35-4b_battery100_ablation.json")
orig = load("battery/qwen35-4b_battery100.json")
fx1 = load("battery/qwen35-4b_battery100_fixed.json")
fx2 = load("battery/qwen35-4b_battery100_fixed2.json")
def q35_all():
    out = {}
    for src in (orig, fx1, fx2, v3, abl, v4vec):
        if not src:
            continue
        sess = next(iter(src["sessions"].values()))
        for name, d in sess["strategies"].items():
            if name in ("v4_rag_kv", "v4_kv_mid", "v3_rag_centroid", "v3_centroid_plus_gist", "classic_kv_full"):
                # prefer latest runs (list order) but keep earliest original for stages 0-1
                out[name] = d
            else:
                out.setdefault(name, d)
    # explicit overrides from the correct (fixed) stages
    if fx2:
        s2 = next(iter(fx2["sessions"].values()))["strategies"]
        out["v2_rag_plain_fixes"] = s2.get("v2_rag_plain")
        out["v2_rag_gist_fixes"] = s2.get("v2_rag_gist")
    return out



# ---------------------------------------------------------------- metrics utils
def rel_tone(d, classic):
    t, c = d.get("tone_mean"), (classic or {}).get("tone_mean")
    return None if (t is None or not c) else t / c


def fmt(v, spec="{:.2f}", dash="—"):
    return dash if v is None else spec.format(v)


# ---------------------------------------------------------------- figures
plt.rcParams.update({"figure.dpi": 140, "font.size": 9})

def fig_evolution(q):
    stages = [
        ("Эталон", q.get("classic_kv_full")),
        ("1: хук\n(ноутбук)", q.get("v1_memory")),
        ("2: банк\n(до фиксов)", q.get("v2_rag_plain")),
        ("2: банк\n+фиксы", q.get("v2_rag_plain_fixes")),
        ("3: центроид", q.get("v3_rag_centroid")),
        ("3: центроид\n+gist", q.get("v3_centroid_plus_gist")),
        ("4: банк+\nattention", q.get("v4_rag_kv")),
        ("4: только\nattention", q.get("v4_kv_mid")),
    ]
    stages = [(n, d) for n, d in stages if d]
    x = np.arange(len(stages))
    recall = [d["recall_mean"] for _, d in stages]
    slots = [d["cache_slots"] / max(q["classic_kv_full"]["cache_slots"], 1) for _, d in stages]
    tone = [rel_tone(d, q["classic_kv_full"]) or np.nan for _, d in stages]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True)
    bars = a1.bar(x, recall, color=["#888"] + ["#4c72b0"] * (len(x) - 1), width=.62)
    a1.bar(0, recall[0], color="#c44e52")
    a1.set_ylabel("Точность фактов, доля вопросов")
    a1.set_ylim(0, 1.05)
    for xi, v in zip(x, recall):
        a1.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    a1.plot(x, tone, "o--", color="#dd8452", label="удержание темы (×эталона)")
    a1.axhline(1.0, color="#dd8452", lw=.6, ls=":")
    a1.set_title("Эволюция системы: точность фактов и удержание темы (Qwen3.5-4B, сессия 107 ходов / 24 вопроса)")
    a2.bar(x, [s * 100 for s in slots], color="#55a868", width=.62)
    for xi, v in zip(x, slots):
        a2.text(xi, v + 1, f"{v * 100:.0f}%", ha="center", fontsize=8)
    a2.set_yscale("log")
    a2.set_ylabel("attention-работа, % эталона")
    a2.set_title("Стоимость (меньше = лучше)")
    a2.set_xticks(x)
    a2.set_xticklabels([n for n, _ in stages], fontsize=8)
    a1.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "evolution.png")
    plt.close(fig)


def fig_pareto(q):
    pts = [
        ("Эталон (полный контекст)", q.get("classic_kv_full"), "#c44e52", "o"),
        ("Буфер без памяти", q.get("window2_only"), "#8c8c8c", "s"),
        ("Ступень 1: хук", q.get("v1_memory"), "#8c8c8c", "^"),
        ("Ступень 2: банк", q.get("v2_rag_plain_fixes") or q.get("v2_rag_plain"), "#4c72b0", "D"),
        ("Ступень 3: центроид", q.get("v3_rag_centroid"), "#55a868", "P"),
        ("Ступень 3: +gist", q.get("v3_centroid_plus_gist"), "#30a2da", "X"),
        ("Ступень 4: банк+attention", q.get("v4_rag_kv"), "#dd8452", "*"),
        ("Скользящий KV-128", q.get("sparse_sliding_128"), "#8172b2", "v"),
        ("Скользящий KV-32", q.get("sparse_sliding_32"), "#8172b2", "."),
        ("KV top-k (H2O)", q.get("sparse_topk_128"), "#937860", "h"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 6))
    for label, d, color, mk in pts:
        if not d or not d.get("cache_slots"):
            continue
        ax.scatter(d["cache_slots"], d["recall_mean"], s=130, color=color, marker=mk, zorder=3)
        ax.annotate(label, (d["cache_slots"], d["recall_mean"]), textcoords="offset points",
                    xytext=(8, 5), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("attention-работа, слоты кэша (лог-шкала; меньше = дешевле)")
    ax.set_ylabel("Точность фактов (выше = лучше)")
    ax.set_title("Парето: качество фактов против стоимости (Qwen3.5-4B, 107 ходов)")
    ax.grid(alpha=.3, ls="--")
    fig.tight_layout()
    fig.savefig(FIGS / "pareto.png")
    plt.close(fig)


def load_bench(model):
    data = load(f"gpu/{model}_benchmark.json")
    if not data:
        return None
    sess = data["sessions"].get("coding-buddy") or next(iter(data["sessions"].values()))
    return {k: v["summary"] for k, v in sess.items()}


def fig_models():
    models = [("gpt2", "GPT-2 (124M)"), ("t5-base", "T5-base (223M)"), ("qwen2-1.5b", "Qwen2-1.5B")]
    cfgs = [("classic_kv_full", "Эталон: полный контекст+KV", "#c44e52"),
            ("sparse_sliding", "Скользящий KV-кэш", "#8172b2"),
            ("sparse_topk", "KV top-k (H2O)", "#937860"),
            ("window_only", "Буфер без памяти", "#8c8c8c"),
            ("window_hybrid_memory", "Буфер + память (ступень 1)", "#4c72b0")]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    width = 0.16
    xs = np.arange(len(models))
    for j, (cfg, label, color) in enumerate(cfgs):
        vals = []
        for m, _ in models:
            b = load_bench(m) or {}
            d = b.get(cfg, {})
            vals.append(d.get("probe_mean") or 0.0)
        ax.bar(xs + (j - 2) * width, vals, width, label=label, color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in models])
    ax.set_ylabel("Точность фактов (доля контрольных вопросов)")
    ax.set_xlabel("Сессия с заложенными фактами (10-11 ходов); сравнение политик кэша и памяти")
    ax.set_title("Базовое сравнение на трёх моделях")
    ax.legend(fontsize=7.5, loc="upper left", ncols=2)
    fig.tight_layout()
    fig.savefig(FIGS / "smallmodels.png")
    plt.close(fig)


def other_model_table():
    tables = {}
    for m in ("gpt2", "t5-base", "qwen2-1.5b"):
        data = load(f"stress/{m}_stress.json")
        if not data:
            continue
        for sname, sess in data["sessions"].items():
            if "long" in sname:
                tables[m] = sess["strategies"]
    return tables


def fig_training():
    runs = {}
    for tag, fname in (("прогон 2 (общие LoRA)", "battery/trainreader_run2.log"),
                       ("прогон 3 (выделенный читатель)", "battery/trainreader_run3.log")):
        p = RES / fname
        if not p.exists():
            continue
        steps, losses = [], []
        for line in p.read_text(errors="ignore").splitlines():
            m = re.match(r".*step (\d+): loss ([\d.]+) eval recall", line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
        if steps:
            runs[tag] = (steps, losses)
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5))
    for (tag, (s, l)), color in zip(runs.items(), ("#4c72b0", "#55a868")):
        ax.plot(s, l, "-o", ms=4, color=color, label=tag)
    ax.set_yscale("log")
    ax.set_xlabel("шаг обучения")
    ax.set_ylabel("loss обучающей последовательности (лог)")
    ax.set_title("Обучение «читателя памяти»: loss падает, точность на свежих кодах остаётся 0/24\n"
                 "Позитивный контроль (факт в промпте) = 1.00 — измеритель исправен")
    ax.grid(alpha=.3, ls="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "training.png")
    plt.close(fig)

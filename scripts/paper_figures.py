"""Figures from the historical Qwen3.5 comparison and current core/RAG corpus series."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import report_data as RD

FIGS = RD.FIGS
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
COLORS = ["#c44e52", "#8172b2", "#937860", "#55a868", "#30a2da", "#dd8452", "#4c72b0"]


def _entries(q):
    return [(RD.LABELS[k], q[k], c) for k, c in zip(RD.ORDER, COLORS) if k in q]


def fig_main(q):
    entries = _entries(q)
    labels = [label.replace(" ", "\n", 1) for label, _, _ in entries]
    x = np.arange(len(entries))
    reference = q["classic_kv_full"]
    panels = [
        ("Точность фактов (выше = лучше)", [d["recall_mean"] for _, d, _ in entries], "{:.2f}", False),
        ("Удержание темы (косинус)", [d["tone_mean"] for _, d, _ in entries], "{:.2f}", False),
        ("Attention-работа, % чистого Qwen (лог-шкала)",
         [d["cache_slots"] / reference["cache_slots"] * 100 for _, d, _ in entries], "{:.0f}%", True),
        ("Наблюденная латентность генерации, с/ход", [d["mean_turn_latency_s"] for _, d, _ in entries], "{:.2f}", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (title, values, spec, log) in zip(axes.flat, panels):
        ax.bar(x, values, color=[c for _, _, c in entries], width=0.65)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(alpha=0.25, ls="--", axis="y")
        if log:
            ax.set_yscale("log")
        span = max(values) - min(values) if values else 1
        for xi, value in zip(x, values):
            y = value * 1.04 if log else value + max(span * 0.025, 0.015)
            ax.text(xi, y, spec.format(value), ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("Qwen3.5-4B · 107 ходов · 24 вопроса · единый прогон")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "comparison.png")
    plt.close(fig)


def fig_pareto(q):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for (label, d, color) in _entries(q):
        ax.scatter(d["cache_slots"], d["recall_mean"], s=150, color=color, zorder=3)
        if label.startswith("Чистый Qwen"):
            ax.annotate(label, (d["cache_slots"], d["recall_mean"]), xytext=(-7, 10),
                        textcoords="offset points", fontsize=8, ha="right")
        else:
            ax.annotate(label, (d["cache_slots"], d["recall_mean"]), xytext=(7, 5),
                        textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.1)
    ax.set_xlabel("Attention-слоты за сессию (лог-шкала; меньше = дешевле)")
    ax.set_ylabel("Recall (больше = лучше)")
    ax.set_title("Качество и стоимость на одной батарее Qwen3.5-4B")
    ax.grid(alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(FIGS / "pareto.png")
    plt.close(fig)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.8, 6.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, color="#4c72b0"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                               fc="#f7f9fc", ec=color, lw=1.4)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, linespacing=1.4)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.2, color="#555"))

    box(.3, 3.0, 2.0, 1.0, "Новая реплика\nпользователя", "#777777")
    box(3.0, 4.8, 3.0, 1.35, "Текстовый retrieval\nЦентрирование e − c\nCosine top-k + upsert", "#55a868")
    box(3.0, 1.1, 3.0, 1.35, "Латентный retrieval\nАктивации h; TTL + EMA\nКосинусная выдача", "#dd8452")
    box(7.0, 4.8, 2.6, 1.35, "Промпт\nБуфер 10 ходов (10 пар)\n+ до 4 заметок", "#8172b2")
    box(7.0, 1.1, 2.3, 1.35, "Замороженный Qwen\na → Kₘ, Vₘ в средних\nattention-слоях", "#4c72b0")
    box(10.0, 3.0, 1.7, 1.0, "Ответ\nмодели", "#c44e52")
    arrow(2.3, 3.65, 3.0, 5.1)
    arrow(2.3, 3.35, 3.0, 2.1)
    arrow(6.0, 5.5, 7.0, 5.5)
    arrow(8.15, 4.8, 8.15, 2.45)
    arrow(6.0, 1.8, 7.0, 1.8)
    arrow(9.3, 2.0, 10.25, 3.0)
    ax.text(6, 6.65, "Двухканальная память: точные факты — текстом, состояние тем — через K/V",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(6, .35, "Веса модели не обучаются; память не добавляется в последовательный KV-кэш",
            ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    fig.savefig(FIGS / "architecture.png")
    plt.close(fig)


def make_all():
    q = RD.q35_all()
    fig_main(q)
    fig_pareto(q)
    fig_architecture()
    make_article_figures()


if __name__ == "__main__":
    make_all()


def make_article_figures():
    """Copy current core/RAG-only figures and render the current hybrid comparison."""
    import shutil

    sources = {
        "core_quality_cost.png": RD.RES / "corpus/full-core-v1/core_quality_cost.png",
        "core_per_dialogue.png": RD.RES / "corpus/full-core-v1/core_per_dialogue.png",
        "core_lag.png": RD.RES / "corpus/full-core-v1/core_lag.png",
        "rag_only_quality_cost.png": RD.RES / "corpus/full-rag-v1/rag_only_quality_cost.png",
        "rag_only_layer_centering.png": RD.RES / "corpus/full-rag-v1/rag_only_layer_centering.png",
        "rag_only_per_dialogue.png": RD.RES / "corpus/full-rag-v1/rag_only_per_dialogue.png",
        "rag_only_lag.png": RD.RES / "corpus/full-rag-v1/rag_only_lag.png",
    }
    for name, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(f"Сначала постройте актуальный рисунок: {source}")
        shutil.copyfile(source, FIGS / name)

    data = RD.rag_data()
    rows = data["pair_rows"]
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    pure_values = [row["pure_recall"] for row in rows]
    dual_values = [row["dual_recall"] for row in rows]
    pure_bars = ax.bar(x - width / 2, pure_values, width, label="RAG без K/V", color="#64748b")
    dual_bars = ax.bar(x + width / 2, dual_values, width, label="RAG + живая K/V (k32)", color="#059669")
    full = data["metrics"]["classic_kv_full"]["recall_mean"]
    ax.axhline(full, color="#b91c1c", linestyle="--", lw=1.4, label=f"Full KV = {full:.3f}")
    ax.set_xticks(x, [row["label"] for row in rows], rotation=22, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Recall")
    ax.set_title(f"Вклад K/V в текстовый RAG · {len(data['records'])} диалога", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for bars in (pure_bars, dual_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7.5)
    ax.legend(ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGS / "rag_hybrid_pairs.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

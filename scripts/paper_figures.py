"""Paper-grade figures: main comparison panels + architecture diagram."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import report_data as RD

FIGS = RD.FIGS

SYSTEMS = [  # (json key, подпись, цвет)
    ("classic_kv_full", "Базовая модель\n(полный контекст + классический KV)", "#c44e52"),
    ("sparse_sliding_128", "Разрежённое внимание\n(скользящий KV-кэш, 128)", "#8172b2"),
    ("sparse_topk_128", "Разрежённое внимание\n(top-k по вниманию, H2O)", "#937860"),
    ("window2_only", "Буфер без памяти\n(наивное удешевление)", "#8c8c8c"),
    ("v4_rag_kv", "Наша система\n(банк + темы в attention)", "#4c72b0"),
]


def fig_paper_main(q):
    stages = [(lbl, q.get(k)) for k, lbl, _ in SYSTEMS]
    stages = [(lbl, d) for lbl, d in stages if d]
    labels = [lbl for lbl, _ in stages]
    x = np.arange(len(stages))
    colors = [c for _, _, c in SYSTEMS if q.get(_)]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    ax = axes[0][0]
    vals = [d["recall_mean"] for _, d in stages]
    ax.bar(x, vals, color=colors, width=.62)
    ax.set_ylim(0, 1.08)
    ax.set_title("а) Точность фактов (доля верно возвращённых фактов)")
    ax.axhline(1, color="#c44e52", lw=.8, ls="--")

    ax = axes[0][1]
    tone = [RD.rel_tone(d, q["classic_kv_full"]) for _, d in stages]
    ax.bar(x, tone, color=colors, width=.62)
    ax.axhline(1.0, color="k", lw=.8, ls=":")
    ax.set_title("б) Удержание темы (нормировано на базовую модель)")

    ax = axes[1][0]
    slots = [d["cache_slots"] for _, d in stages]
    ax.bar(x, slots, color=colors, width=.62)
    ax.set_yscale("log")
    ax.set_title("в) Attention-работа (слоты кэша, лог-шкала)")

    ax = axes[1][1]
    lat = [d["mean_turn_latency_s"] for _, d in stages]
    ax.bar(x, lat, color=colors, width=.62)
    ax.set_title("г) Латентность хода, с")

    for row in axes:
        for a in row:
            a.set_xticks(x)
            a.set_xticklabels(labels, fontsize=7)
            a.grid(alpha=.25, ls="--", axis="y")
    fig.suptitle("Qwen3.5-4B · диалог 107 ходов · 24 контрольных вопроса (каждый второй — переформулированный)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "paper_main.png")
    plt.close(fig)



def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.8, 7.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec="#4c72b0", fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                linespacing=1.35, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color="#444", style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                     color=color, lw=1.15))

    ax.text(6, 8.3, "Двухслойная память: текстовые ОТКЛОНЕНИЯ от темы — в промпте; скрытые темы — ключами/значениями внимания",
            ha="center", fontsize=10.5, fontweight="bold")

    # левая колонка: поток и состояние
    box(0.25, 6.6, 2.7, 0.9, "Поток диалога\n(реплики пользователя)")
    box(0.25, 5.1, 2.7, 0.9, "Эмбеддинги реплик\ne\u1d62 = pool(f\u02b0(H))", "#f4f6fa")
    box(0.25, 3.6, 2.7, 0.9, "Тема диалога — центроид\nc\u209c (бегущее среднее)", "#fdf3e3", "#dd8452")
    box(0.25, 1.7, 2.7, 1.3, "Хранилище ОТКЛОНЕНИЙ\nu\u1d62 = e\u1d62 \u2212 c  + текст факта\n(сырые эмбеддинги не хранятся)", "#eaf7ec", "#55a868")
    arrow(1.6, 6.6, 1.6, 6.0)
    arrow(1.6, 5.1, 1.6, 4.5)
    arrow(1.6, 3.6, 1.6, 3.0)

    # средняя колонка: банк тем и операции
    box(3.35, 5.1, 2.9, 0.9, "Банк тем (скрытое состояние)\nслоты с TTL, EMA a\u2190(1\u2212k)a+k\u00b7b", "#fdf3e3", "#dd8452")
    box(3.35, 3.6, 2.9, 1.1, "Upsert-дедуп:\ncos(u\u2099,u\u2096) \u2265 \u03b8 \u21d2 заменить текст\n(смена факта побеждает старое)", "#eaf7ec", "#55a868")
    box(3.35, 1.7, 2.9, 1.3, "Контекстуализованный запрос\nq\u2032 = norm(q \u2212 c + \u03bb\u00b7gist)\nвыдача: топ-k заметок ТЕКСТОМ", "#eaf7ec", "#55a868")
    arrow(3.35, 5.55, 2.95, 4.3, "#dd8452", "<->")
    arrow(3.0, 2.35, 3.35, 2.35)
    arrow(4.8, 3.6, 4.8, 3.0)
    arrow(6.25, 2.35, 7.35, 6.05, "#55a868")

    # правая колонка: модель
    box(7.4, 6.3, 4.3, 0.9, "ПРОМПТ = буфер свежих 2\u20135 ходов\n+ k заметок из банка", "#f6eefb", "#8172b2", bold=True)
    arrow(3.0, 7.05, 7.4, 7.05)
    box(7.4, 1.15, 4.3, 4.6, "", "#f7f7fb", "#999")
    ax.text(9.55, 5.4, "Трансформер (замораживается)", ha="center", fontsize=9, fontweight="bold")
    box(7.6, 4.2, 3.9, 0.7, "ранние слои: лексика", "#ffffff", "#c8c8c8")
    box(7.6, 2.55, 3.9, 1.25, "СРЕДНИЕ attention-слои:\nK,V \u2190 [K,V ; K\u2098,V\u2098] без позиции (RoPE \u2298)\nвыход = Attn(Q,K,V) + tanh(g)\u00b7Attn(Q,K\u2098,V\u2098)\ng инициализирован нулём", "#fdf3e3", "#dd8452", fs=8)
    box(7.6, 1.45, 3.9, 0.7, "поздние слои: задача/вывод", "#ffffff", "#c8c8c8")
    arrow(9.55, 6.3, 9.55, 5.85)
    arrow(6.25, 5.1, 7.6, 3.35, "#dd8452")
    ax.text(6.95, 4.5, "строки-темы\n(безпозиционные K/V)", fontsize=7.5, color="#dd8452", ha="center")

    ax.text(6, 0.55, "Смысловое состояние диалога \u2014 RNN-аналог: TTL \u03c4\u2190min(6, \u03c4+2) при попадании и \u03c4\u22121 на каждый запрос; "
                     "отклонения u=e\u2212c убирают анизотропию эмбеддингов;\nканал внимания по GPU-стоимости \u2248 0 (32 ключа к ~3000), "
                     "всю цену даёт необученный внешний retrieval",
            ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(FIGS / "architecture.png")
    plt.close(fig)


if __name__ == "__main__":
    q = RD.q35_all()
    fig_paper_main(q)
    fig_architecture()
    print("paper figures ok")

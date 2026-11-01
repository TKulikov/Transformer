"""Paper-grade figures, v2: self-explanatory panels, no overlaps.

All bar charts carry numeric labels; every axis states the metric and its
direction; systems are compared against the full-context reference (=1.0).
"""

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

plt.rcParams.update({"figure.dpi": 150, "font.size": 9})

COLORS = {"ref": "#c44e52", "kv": "#8172b2", "buf": "#8c8c8c", "ours": "#4c72b0", "ours2": "#55a868"}


def _systems(q):
    spec = [
        ("classic_kv_full", "Эталон:\nполный контекст", COLORS["ref"]),
        ("sparse_sliding_128", "Конкурент:\nскользящий KV-128", COLORS["kv"]),
        ("sparse_topk_128", "Конкурент:\nKV top-k (H2O)", "#937860"),
        ("window2_only", "Нижняя граница:\nбуфер без памяти", COLORS["buf"]),
        ("v4_rag_kv", "НАША СИСТЕМА\n(финальная)", COLORS["ours"]),
    ]
    return [(lbl, q[k], c) for k, lbl, c in spec if q.get(k)]


def fig_paper_main(q):
    entries = _systems(q)
    x = np.arange(len(entries))
    ref = q["classic_kv_full"]
    panels = [
        ("Точность фактов\n(доля верно возвращённых; выше = лучше)",
         [d["recall_mean"] for _, d, _ in entries], "{:.2f}", False, (0, 1.12)),
        ("Удержание темы\n(нормировано на эталон = 1.00)",
         [RD.rel_tone(d, ref) or 0 for _, d, _ in entries], "{:.2f}", False, (0.9, 1.28)),
        ("Стоимость внимания\n(attention-слоты, % от эталона; лог-шкала)",
         [d["cache_slots"] / ref["cache_slots"] * 100 for _, d, _ in entries], "{:.0f}%", True, None),
        ("Латентность хода, с\n(меньше = быстрее)",
         [d["mean_turn_latency_s"] for _, d, _ in entries], "{:.2f}", False, (0, 2.35)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6))
    for ax, (title, vals, spec, log, ylim) in zip(axes.flat, panels):
        ax.bar(x, vals, color=[c for _, _, c in entries], width=0.62)
        ax.set_title(title, fontsize=9.5)
        if log:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        for xi, v in zip(x, vals):
            top = v * (1.04 if log else 1.0) + (0.0 if log else (ylim[1] - ylim[0]) * 0.03 if ylim else max(vals) * 0.04)
            ax.text(xi, top, spec.format(v), ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for lbl, _, _ in entries], fontsize=7.5)
        ax.grid(alpha=0.25, ls="--", axis="y")
    fig.suptitle("Qwen3.5-4B · 107 ходов · 24 вопроса\n"
                 "красный — эталон, фиолетовый/серый — аналоги из литературы, зелёный/синий — наша система",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGS / "paper_main.png")
    plt.close(fig)


def fig_evolution(q):
    spec = [
        ("classic_kv_full", "Эталон"),
        ("v1_memory", "1: хук\n(ноутбук)"),
        ("v2_rag_plain", "2: банк\n(до фиксов)"),
        ("v2_rag_plain_fixes", "2: банк\n+фиксы"),
        ("v3_rag_centroid", "3: центроид"),
        ("v3_centroid_plus_gist", "3: центроид\n+gist"),
        ("v4_rag_kv", "4: банк+\nattention"),
        ("v4_kv_mid", "4: только\nattention"),
    ]
    stages = [(lbl, q[k]) for k, lbl in spec if q.get(k)]
    x = np.arange(len(stages))
    ref = q["classic_kv_full"]
    recall = [d["recall_mean"] for _, d in stages]
    tone = [RD.rel_tone(d, ref) or np.nan for _, d in stages]
    cost = [d["cache_slots"] / ref["cache_slots"] * 100 for _, d in stages]

    fig, axes = plt.subplots(3, 1, figsize=(10.6, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1, 1]})
    a1, a2, a3 = axes
    a1.bar(x, recall, color=[COLORS["ref"]] + [COLORS["ours"]] * (len(x) - 1), width=0.6)
    a1.set_ylim(0, 1.15)
    a1.set_ylabel("Точность фактов\n(доля вопросов)")
    for xi, v in zip(x, recall):
        a1.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
    a1.set_title("Эволюция системы 1→4 (Qwen3.5-4B, батарея 107 ходов / 24 вопроса)")

    a2.plot(x, tone, "o-", color="#dd8452", lw=1.6, ms=5)
    a2.axhline(1.0, color="k", lw=0.7, ls=":")
    a2.text(x[-1] + 0.15, 1.0, "эталон=1", fontsize=7, va="center")
    a2.set_ylim(0.9, 1.25)
    a2.set_ylabel("Удержание темы\n(×эталона)")
    for xi, v in zip(x, tone):
        a2.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

    bars = a3.bar(x, cost, color=COLORS["ours2"], width=0.6)
    a3.set_yscale("log")
    a3.set_ylim(5, 160)
    a3.set_ylabel("Стоимость внимания\n(% эталона, лог)")
    a3.set_xlabel("меньше столбик = дешевле;  эталон = 100 %")
    for xi, v in zip(x, cost):
        a3.text(xi, v * 1.08, f"{v:.0f}%", ha="center", fontsize=7.5)
    a3.set_xticks(x)
    a3.set_xticklabels([lbl for lbl, _ in stages], fontsize=8)
    for a in axes:
        a.grid(alpha=0.25, ls="--", axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "evolution.png")
    plt.close(fig)


_PARETO_OFFSETS = {
    "Эталон (полный контекст)": (-140, 10),
    "Скользящий KV-128": (12, -4),
    "KV top-k (H2O)": (12, -14),
    "Буфер без памяти": (-40, 16),
    "Ступень 1: хук": (-150, 22),          # подпись-двойка: см. примечание ниже
    "Ступень 2: банк": (14, -2),
    "Ступень 3: центроид": (-40, -26),
    "Ступень 3: +gist": (12, 10),
    "Ступень 4: банк+attention": (12, 8),
    "Ступень 4: только attention": None,   # совпадает с буфером/хуком — не подписываем отдельно
}


def fig_pareto(q):
    pts = [
        ("Эталон (полный контекст)", "classic_kv_full", COLORS["ref"], "o"),
        ("Скользящий KV-128", "sparse_sliding_128", COLORS["kv"], "v"),
        ("KV top-k (H2O)", "sparse_topk_128", "#937860", "h"),
        ("Буфер без памяти", "window2_only", COLORS["buf"], "s"),
        ("Наша система", "v4_rag_kv", COLORS["ours"], "*"),
    ]
    fig, ax = plt.subplots(figsize=(10.6, 6.6))
    offsets = {"Эталон (полный контекст)": (-140, 10), "Скользящий KV-128": (12, -4),
               "KV top-k (H2O)": (12, -16), "Буфер без памяти": (12, 10), "Наша система": (12, 8)}
    for label, key, color, mk in pts:
        d = q.get(key)
        if not d:
            continue
        ax.scatter(d["cache_slots"], d["recall_mean"], s=170, color=color, marker=mk, zorder=3)
        dx, dy = offsets[label]
        ax.annotate(label, (d["cache_slots"], d["recall_mean"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8.5)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("Стоимость: attention-слоты за сессию (лог-шкала; влево = дешевле)")
    ax.set_ylabel("Точность фактов (вверх = лучше)")
    ax.set_title("Парето: качество фактов против стоимости внимания (Qwen3.5-4B, 107 ходов)\n"
                 "Идеальная система — верхний левый угол; эталон — вверху справа", fontsize=10)
    ax.grid(alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(FIGS / "pareto.png")
    plt.close(fig)


def fig_smallmodels():
    models = [("gpt2", "GPT-2 (124M)"), ("t5-base", "T5-base (223M)"), ("qwen2-1.5b", "Qwen2-1.5B")]
    cfgs = [("classic_kv_full", "Эталон: полный контекст+KV", COLORS["ref"]),
            ("sparse_sliding", "Скользящий KV-кэш", COLORS["kv"]),
            ("sparse_topk", "KV top-k (H2O)", "#937860"),
            ("window_only", "Буфер без памяти", COLORS["buf"]),
            ("window_hybrid_memory", "Буфер + память (ступень 1)", COLORS["ours"])]
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    width = 0.16
    xs = np.arange(len(models))
    for j, (cfg, label, color) in enumerate(cfgs):
        vals = []
        for m, _ in models:
            b = RD.load_bench(m) or {}
            vals.append((b.get(cfg) or {}).get("probe_mean") or 0.0)
        pos = xs + (j - 2) * width
        ax.bar(pos, vals, width, label=label, color=color)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in models])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Точность фактов (доля верно возвращённых)")
    ax.set_title("Базовое сравнение политик на малых моделях (сессии с заложенными фактами)")
    ax.legend(fontsize=8, loc="upper left", ncols=2)
    ax.grid(alpha=0.25, ls="--", axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "smallmodels.png")
    plt.close(fig)


def fig_training():
    import re
    runs = [("прогон 2: LoRA на общих проекциях (3.1M)", "battery/trainreader_run2.log", COLORS["ours"]),
            ("прогон 3: выделенные головы с нулевым гейтом (45.1M)", "battery/trainreader_run3.log", COLORS["ours2"])]
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    for tag, fname, color in runs:
        p = RD.RES / fname
        if not p.exists():
            continue
        steps, losses = [], []
        for line in p.read_text(errors="ignore").splitlines():
            m = re.match(r".*step (\d+): loss ([\d.]+) eval recall ([\d.]+)", line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
        if steps:
            ax.plot(steps, losses, "-o", ms=4, color=color, label=f"{tag} — loss")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 10)
    ax.axhline(1.0, color=COLORS["ref"], ls="--", lw=1)
    ax.text(50, 0.62, "позитивный контроль: факт в промпте → точность 1.00 (измеритель исправен)",
            fontsize=8, color=COLORS["ref"])
    ax.text(600, 1.4e-3, "точность на свежих кодах: 0.00 у обоих прогонов (линия по нулям, не показана)",
            fontsize=8, color="#555", ha="right")
    ax.set_xlabel("шаг обучения")
    ax.set_ylabel("loss обучающей последовательности (лог-шкала)")
    ax.set_title("Попытка обучить модель читать память (негативный результат):\n"
                 "loss падает, а точность на невиданных ранее фактах остаётся нулевой", fontsize=10)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "training.png")
    plt.close(fig)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.8, 7.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec="#4c72b0", fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                linespacing=1.35, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color="#444", style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                     color=color, lw=1.15))

    ax.text(6, 8.3, "Двухслойная память: текстовые ОТКЛОНЕНИЯ от темы — в промпте; скрытые темы — ключами/значениями внимания",
            ha="center", fontsize=10.5, fontweight="bold")

    box(0.25, 6.6, 2.7, 0.9, "Поток диалога\n(реплики пользователя)")
    box(0.25, 5.1, 2.7, 0.9, "Эмбеддинги реплик\ne\u1d62 = pool(f\u02b0(H))", "#f4f6fa")
    box(0.25, 3.6, 2.7, 0.9, "Тема диалога — центроид\nc\u209c (бегущее среднее)", "#fdf3e3", "#dd8452")
    box(0.25, 1.7, 2.7, 1.3, "Хранилище ОТКЛОНЕНИЙ\nu\u1d62 = e\u1d62 \u2212 c  + текст факта\n(сырые эмбеддинги не хранятся)", "#eaf7ec", "#55a868")
    arrow(1.6, 6.6, 1.6, 6.0)
    arrow(1.6, 5.1, 1.6, 4.5)
    arrow(1.6, 3.6, 1.6, 3.0)

    box(3.35, 5.1, 2.9, 0.9, "Банк тем (скрытое состояние)\nслоты с TTL, EMA a\u2190(1\u2212k)a+k\u00b7b", "#fdf3e3", "#dd8452")
    box(3.35, 3.6, 2.9, 1.1, "Upsert-дедуп:\ncos(u\u2099,u\u2096) \u2265 \u03b8 \u21d2 заменить текст\n(смена факта побеждает старое)", "#eaf7ec", "#55a868")
    box(3.35, 1.7, 2.9, 1.3, "Контекстуализованный запрос\nq\u2032 = norm(q \u2212 c + \u03bb\u00b7gist)\nвыдача: топ-k заметок ТЕКСТОМ", "#eaf7ec", "#55a868")
    arrow(3.35, 5.55, 2.95, 4.3, "#dd8452", "<->")
    arrow(3.0, 2.35, 3.35, 2.35)
    arrow(4.8, 3.6, 4.8, 3.0)
    arrow(6.25, 2.35, 7.5, 6.2, "#55a868")

    box(7.4, 6.3, 4.3, 0.9, "ПРОМПТ = буфер свежих 2\u20135 ходов\n+ k заметок из банка", "#f6eefb", "#8172b2", bold=True)
    arrow(3.0, 7.05, 7.4, 7.05)
    box(7.4, 1.15, 4.3, 4.6, "", "#f7f7fb", "#999")
    ax.text(9.55, 5.4, "Трансформер (замораживается)", ha="center", fontsize=9, fontweight="bold")
    box(7.6, 4.2, 3.9, 0.7, "ранние слои: лексика", "#ffffff", "#c8c8c8")
    box(7.6, 2.55, 3.9, 1.25, "СРЕДНИЕ attention-слои:\nK,V \u2190 [K,V ; K\u2098,V\u2098] без позиции (RoPE \u2298)\nвыход = Attn(Q,K,V) + tanh(g)\u00b7Attn(Q,K\u2098,V\u2098)\ng инициализирован нулём", "#fdf3e3", "#dd8452", fs=8)
    box(7.6, 1.45, 3.9, 0.7, "поздние слои: задача/вывод", "#ffffff", "#c8c8c8")
    arrow(9.55, 6.3, 9.55, 5.85)
    arrow(6.25, 5.35, 7.6, 3.5, "#dd8452")
    ax.text(6.05, 4.05, "строки-темы\n(безпозиционные K/V)", fontsize=7.5, color="#dd8452", ha="center")

    ax.text(6, 0.55, "Смысловое состояние диалога \u2014 RNN-аналог: TTL \u03c4\u2190min(6, \u03c4+2) при попадании и \u03c4\u22121 на каждый запрос; "
                     "отклонения u=e\u2212c убирают анизотропию эмбеддингов;\nканал внимания по GPU-стоимости \u2248 0 "
                     "(32 ключа к ~3000), всю цену даёт необученный внешний retrieval",
            ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(FIGS / "architecture.png")
    plt.close(fig)


def make_all():
    q = RD.q35_all()
    fig_paper_main(q)
    fig_pareto(q)
    fig_smallmodels()
    fig_training()
    fig_architecture()
    print("paper figures v3 ok")


if __name__ == "__main__":
    make_all()

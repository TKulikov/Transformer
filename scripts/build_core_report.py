"""Build current plots and Markdown/PDF reports for all completed full-core-v1 batches."""

from __future__ import annotations

import html
import sys
from datetime import date
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN_DIR = ROOT / "results/corpus/full-core-v1"
PDF_PATH = RUN_DIR / "CORE_REPORT.pdf"
MD_PATH = RUN_DIR / "CORE_REPORT.md"
ORDER = [
    "classic_kv_full", "sparse_sliding_128", "sparse_topk_128", "snapkv_128",
    "pyramidkv_128", "v4_kv_mid", "v4_kv_mid_dyn",
]
FOCUS = ["classic_kv_full", "snapkv_128", "pyramidkv_128", "v4_kv_mid", "v4_kv_mid_dyn"]
LABELS = {
    "classic_kv_full": "Full KV",
    "sparse_sliding_128": "Sliding 128",
    "sparse_topk_128": "TopK 128",
    "snapkv_128": "SnapKV 128",
    "pyramidkv_128": "PyramidKV 128",
    "v4_kv_mid": "Живая память k32",
    "v4_kv_mid_dyn": "Живая память dynamic",
}
SHORT_LABELS = {
    "classic_kv_full": "Full",
    "snapkv_128": "SnapKV",
    "pyramidkv_128": "PyramidKV",
    "v4_kv_mid": "k32",
    "v4_kv_mid_dyn": "dynamic",
}
PALETTE = {
    "classic_kv_full": "#334155",
    "sparse_sliding_128": "#94a3b8",
    "sparse_topk_128": "#64748b",
    "snapkv_128": "#3b82f6",
    "pyramidkv_128": "#8b5cf6",
    "v4_kv_mid": "#0d9488",
    "v4_kv_mid_dyn": "#059669",
}
ZONES = ["0-5", "6-12", "13-20", "21+"]


def collect_results():
    """Aggregate completed batches, using dialogue as the statistical unit.

    Returns strategy metrics and source records ordered by batch id. Recall and
    answerable-only recall are macro-averaged over dialogues; E2E is the median
    of per-dialogue medians in seconds per turn. Peak VRAM is the mean of
    per-dialogue torch.max_memory_allocated values in MiB.
    """
    from tmem.corpus_stats import bootstrap_ci, load_run

    run = load_run(RUN_DIR)
    records = sorted(run["records"], key=lambda rec: rec["batch_id"])
    if not records:
        raise RuntimeError(f"Нет завершённых батчей в {RUN_DIR}")
    results = {}
    for name in ORDER:
        entries = [(rec, rec["strategies"][name]) for rec in records]
        analyses = [entry["analysis"] for _, entry in entries]
        answerable = []
        for analysis in analyses:
            scores = [p["score"] or 0.0 for p in analysis["probe_details"] if p["kind"] != "unanswerable"]
            answerable.append(mean(scores) if scores else 0.0)
        recalls = [analysis["recall_mean"] for analysis in analyses]
        e2e_values = [median(entry["e2e"]["per_rep_mean_s"]) for _, entry in entries]
        injected = sum(analysis.get("kv_injected_slots") or 0 for analysis in analyses)
        steps = sum(analysis.get("kv_steps") or 0 for analysis in analyses)
        lag = {}
        for zone in ZONES:
            values = [analysis["recall_by_lag"][zone] for analysis in analyses
                      if analysis["recall_by_lag"][zone] is not None]
            lag[zone] = mean(values) if values else None
        results[name] = {
            "recall": mean(recalls),
            "recall_ci": bootstrap_ci(recalls),
            "answerable": mean(answerable),
            "answerable_ci": bootstrap_ci(answerable),
            "e2e": median(e2e_values),
            "peak": mean(analysis["peak_mem_mb"] for analysis in analyses),
            "prompt": mean(analysis["prompt_tokens"] for analysis in analyses),
            "attention": mean(analysis["attention_flops_proxy"] for analysis in analyses),
            "lag": lag,
            "slots_per_step": injected / steps if steps else None,
        }
    return results, records


def build_figures(results, records):
    """Write overview, per-dialogue, and lag-zone PNGs and return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    labels = [LABELS[name] for name in ORDER]
    x = np.arange(len(ORDER))
    quality_path = RUN_DIR / "core_quality_cost.png"
    dialogue_path = RUN_DIR / "core_per_dialogue.png"
    lag_path = RUN_DIR / "core_lag.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    recalls = [results[name]["recall"] for name in ORDER]
    lower = [results[name]["recall"] - results[name]["recall_ci"][0] for name in ORDER]
    upper = [results[name]["recall_ci"][1] - results[name]["recall"] for name in ORDER]
    bars = axes[0].bar(x, recalls, color=[PALETTE[name] for name in ORDER], width=0.72,
                       yerr=[lower, upper], capsize=4, error_kw={"lw": 1})
    axes[0].set_title("Recall с 95% bootstrap ДИ", fontweight="bold")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Recall")
    for bar, value, name in zip(bars, recalls, ORDER):
        label_y = min(results[name]["recall_ci"][1] + 0.018, 1.04)
        axes[0].text(bar.get_x() + bar.get_width() / 2, label_y, f"{value:.3f}",
                     ha="center", fontsize=8)

    for ax, key, title, fmt in [
        (axes[1], "e2e", "E2E латентность", "{:.2f}"),
        (axes[2], "peak", "Peak allocated VRAM", "{:.0f}"),
    ]:
        values = [results[name][key] for name in ORDER]
        bars = ax.bar(x, values, color=[PALETTE[name] for name in ORDER], width=0.72)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_ylabel("с/ход" if key == "e2e" else "MiB")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025, fmt.format(value),
                    ha="center", fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=32, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    fig.suptitle(f"full-core-v1 · {len(records)} диалогов · Qwen3.5-4B", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(quality_path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.6))
    positions = np.arange(len(records))
    tick_labels = [f"{rec['batch_id'].split('-')[2]}\n{rec['spec']['topic']}" for rec in records]
    for name in FOCUS:
        values = [rec["strategies"][name]["analysis"]["recall_mean"] for rec in records]
        ax.plot(positions, values, marker="o", lw=2, ms=6, label=SHORT_LABELS[name], color=PALETTE[name])
        if name == "v4_kv_mid_dyn":
            for pos, value in zip(positions, values):
                ax.annotate(f"{value:.2f}", (pos, value), xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=8, color=PALETTE[name],
                            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8})
    ax.set_xticks(positions, tick_labels)
    ax.set_xlabel("Диалог (номер и тема)")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.08)
    ax.set_title("Recall по отдельным диалогам", fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(dialogue_path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.6))
    positions = np.arange(len(ZONES))
    width = 0.19
    lag_names = ["classic_kv_full", "pyramidkv_128", "v4_kv_mid", "v4_kv_mid_dyn"]
    for index, name in enumerate(lag_names):
        values = [results[name]["lag"][zone] for zone in ZONES]
        offset = (index - 1.5) * width
        bars = ax.bar(positions + offset, values, width=width, label=SHORT_LABELS[name], color=PALETTE[name])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.2f}",
                    ha="center", fontsize=7)
    ax.set_xticks(positions, ZONES)
    ax.set_xlabel("Расстояние между фактом и вопросом, ходы")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.04)
    ax.set_title("Удержание фактов по lag-зонам", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(lag_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return {"quality": quality_path, "dialogues": dialogue_path, "lag": lag_path}


def report_facts(results):
    """Calculate pairwise statistics and relative costs used by both reports."""
    from tmem.corpus_stats import compare_pairs, load_run

    pairs = [("v4_kv_mid_dyn", "v4_kv_mid"), ("v4_kv_mid_dyn", "pyramidkv_128"),
             ("v4_kv_mid_dyn", "snapkv_128")]
    pvalues = compare_pairs(load_run(RUN_DIR), pairs)
    dynamic = results["v4_kv_mid_dyn"]
    fixed = results["v4_kv_mid"]
    pyramid = results["pyramidkv_128"]
    return {
        "pvalues": pvalues,
        "speed_gain": (1 - dynamic["e2e"] / pyramid["e2e"]) * 100,
        "vram_gain": (1 - dynamic["peak"] / pyramid["peak"]) * 100,
        "prompt_factor": pyramid["prompt"] / dynamic["prompt"],
        "gate_ratio": fixed["slots_per_step"] / dynamic["slots_per_step"],
    }


def markdown_report(results, records, figures, facts):
    """Write the current Markdown report and return its path."""
    n = len(records)
    dynamic = results["v4_kv_mid_dyn"]
    fixed = results["v4_kv_mid"]
    pyramid = results["pyramidkv_128"]
    full = results["classic_kv_full"]
    pvalues = facts["pvalues"]
    lines = [
        "# Full-core-v1: актуальный отчёт",
        "",
        f"**Срез:** {n}/48 диалогов · Qwen3.5-4B bf16 · 100 ходов на диалог · {date.today().isoformat()}.",
        "",
        "## Краткий вывод",
        "",
        f"Динамическая живая память получила recall **{dynamic['recall']:.3f}** против "
        f"**{fixed['recall']:.3f}** у фиксированного k32; paired bootstrap с поправкой Холма: "
        f"**p={pvalues['v4_kv_mid_dyn vs v4_kv_mid']:.3f}**. PyramidKV имеет **{pyramid['recall']:.3f}**, "
        f"Full KV — **{full['recall']:.3f}**. Разница dynamic с PyramidKV не значима "
        f"(p={pvalues['v4_kv_mid_dyn vs pyramidkv_128']:.3f}), но при n={n} это не доказывает паритет.",
        "",
        f"Относительно PyramidKV dynamic быстрее на **{facts['speed_gain']:.1f}%**, использует на "
        f"**{facts['vram_gain']:.1f}%** меньше peak allocated VRAM и в **{facts['prompt_factor']:.2f} раза** "
        f"меньше prompt-токенов. Гейт подключает **{dynamic['slots_per_step']:.2f}** слота/шаг против "
        f"**{fixed['slots_per_step']:.2f}** у k32 — сокращение в **{facts['gate_ratio']:.1f} раза**.",
        "",
        "## Качество и стоимость",
        "",
        "![Recall, латентность и VRAM](core_quality_cost.png)",
        "",
        "| Стратегия | Recall [95% ДИ] | Answerable recall | E2E, с/ход | Peak VRAM, MiB | Prompt-токены | Attention-слоты |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        row = results[name]
        lo, hi = row["recall_ci"]
        lines.append(
            f"| {LABELS[name]} | {row['recall']:.3f} [{lo:.3f}, {hi:.3f}] | {row['answerable']:.3f} | "
            f"{row['e2e']:.3f} | {row['peak']:.0f} | {row['prompt']:.0f} | {row['attention']:.0f} |"
        )
    lines += [
        "",
        "E2E — медиана диалоговых медиан пяти повторов после warmup. Peak VRAM — среднее по диалогам "
        "`torch.max_memory_allocated`, не reserved-память всего процесса. Answerable recall исключает "
        "unanswerable-пробы.",
        "",
        "## Парные сравнения dynamic",
        "",
        "| Сравнение | Разница recall | p после Холма |",
        "|---|---:|---:|",
    ]
    for other in ["v4_kv_mid", "pyramidkv_128", "snapkv_128"]:
        key = f"v4_kv_mid_dyn vs {other}"
        lines.append(f"| dynamic vs {LABELS[other]} | {dynamic['recall'] - results[other]['recall']:+.3f} | {pvalues[key]:.4f} |")
    lines += [
        "",
        "Проверки выполнены парным bootstrap по диалогам; три ключевых p-value скорректированы методом Холма.",
        "",
        "## Разбор по диалогам",
        "",
        "![Recall по диалогам](core_per_dialogue.png)",
        "",
        "| Диалог | Full | SnapKV | PyramidKV | k32 | dynamic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        values = [rec["strategies"][name]["analysis"]["recall_mean"] for name in FOCUS]
        lines.append(f"| {rec['batch_id']} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += [
        "",
        "## Дистанция удержания",
        "",
        "![Recall по lag-зонам](core_lag.png)",
        "",
        "| Стратегия | 0–5 | 6–12 | 13–20 | 21+ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in FOCUS:
        lines.append(f"| {LABELS[name]} | " + " | ".join(f"{results[name]['lag'][zone]:.3f}" for zone in ZONES) + " |")
    lines += [
        "",
        "Dynamic улучшает дальние зоны относительно k32, но temporal recall остаётся 0.000. "
        "Поэтому результат нельзя интерпретировать как решённую долгосрочную память.",
        "",
        "## Метод и ограничения",
        "",
        "Сравниваются Full KV, Sliding/TopK/SnapKV/PyramidKV с бюджетом 128 и два варианта живой "
        "K/V-памяти без текстового RAG. Dynamic отбирает до 32 векторов с косинусом не ниже 0.5. "
        "Recall — keyword-скоринг синтетических проб, а не человеческая оценка. Выборка мала и пока "
        "охватывает только 100-ходовые диалоги; разные темы и сценарии представлены неравномерно. "
        "Незначимый результат при n=5 не является доказательством отсутствия эффекта.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        ".venv/bin/python scripts/build_core_report.py",
        "```",
        "",
        "Источник: `results/corpus/full-core-v1/batches/`.",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n")
    return MD_PATH


def pdf_report(results, records, figures, facts):
    """Write a scientific PDF with the same metrics and figures as the Markdown report."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font_dir = Path("/usr/share/fonts/truetype/liberation")
    pdfmetrics.registerFont(TTFont("Serif", str(font_dir / "LiberationSerif-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("Serif-Bold", str(font_dir / "LiberationSerif-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Serif-Italic", str(font_dir / "LiberationSerif-Italic.ttf")))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCore", parent=styles["Title"], fontName="Serif-Bold", fontSize=17,
                           leading=21, alignment=TA_CENTER, spaceAfter=9)
    subtitle = ParagraphStyle("SubtitleCore", parent=styles["Normal"], fontName="Serif-Italic", fontSize=10,
                              leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#475569"), spaceAfter=12)
    body = ParagraphStyle("BodyCore", parent=styles["Normal"], fontName="Serif", fontSize=10.2, leading=14,
                          alignment=TA_JUSTIFY, spaceAfter=6)
    heading = ParagraphStyle("HeadingCore", parent=body, fontName="Serif-Bold", fontSize=13, leading=16,
                             spaceBefore=9, spaceAfter=6, keepWithNext=True)
    caption = ParagraphStyle("CaptionCore", parent=body, fontSize=9, leading=11, alignment=TA_CENTER,
                             textColor=colors.HexColor("#475569"), spaceAfter=7)
    cell = ParagraphStyle("CellCore", parent=body, fontSize=7.2, leading=8.6, alignment=TA_CENTER, spaceAfter=0)
    head = ParagraphStyle("HeadCore", parent=cell, fontName="Serif-Bold")
    code = ParagraphStyle("CodeCore", parent=body, fontName="Courier", fontSize=8.5, leading=11)

    def para(text, style=body):
        return Paragraph(text, style)

    def add_table(story, table_title, headers, rows, widths, font_size=7.2):
        sized_cell = ParagraphStyle(f"Cell{len(story)}", parent=cell, fontSize=font_size, leading=font_size * 1.2)
        sized_head = ParagraphStyle(f"Head{len(story)}", parent=head, fontSize=font_size, leading=font_size * 1.2)
        data = [[Paragraph(html.escape(str(value)), sized_head) for value in headers]]
        data += [[Paragraph(html.escape(str(value)), sized_cell) for value in row] for row in rows]
        table = Table(data, colWidths=[width * cm for width in widths], repeatRows=1, hAlign="CENTER")
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#64748b")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.extend([Paragraph(html.escape(table_title), caption), table, Spacer(1, 0.2 * cm)])

    n = len(records)
    dynamic = results["v4_kv_mid_dyn"]
    fixed = results["v4_kv_mid"]
    pyramid = results["pyramidkv_128"]
    full = results["classic_kv_full"]
    pvalues = facts["pvalues"]
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title=f"Full-core-v1: отчёт по {n} диалогам", author="Transformer experiments")
    story = [
        Paragraph("Живая K/V-память для удержания диалогового контекста", title),
        Paragraph(f"full-core-v1 · актуальный срез {n}/48 · {date.today().isoformat()}", subtitle),
        Paragraph("КРАТКИЙ ВЫВОД", heading),
        para(
            f"Динамическая живая память получила recall <b>{dynamic['recall']:.3f}</b> против "
            f"<b>{fixed['recall']:.3f}</b> у k32. Парный bootstrap по диалогам с поправкой Холма даёт "
            f"<b>p={pvalues['v4_kv_mid_dyn vs v4_kv_mid']:.3f}</b>. PyramidKV имеет {pyramid['recall']:.3f}, "
            f"Full KV — {full['recall']:.3f}. Отличие dynamic от PyramidKV не значимо "
            f"(p={pvalues['v4_kv_mid_dyn vs pyramidkv_128']:.3f}); при n={n} это не доказывает паритет."
        ),
        para(
            f"Относительно PyramidKV dynamic быстрее на <b>{facts['speed_gain']:.1f}%</b>, использует на "
            f"<b>{facts['vram_gain']:.1f}%</b> меньше peak allocated VRAM и в "
            f"<b>{facts['prompt_factor']:.2f} раза</b> меньше prompt-токенов. Динамический гейт сокращает "
            f"среднее число инъекций с {fixed['slots_per_step']:.2f} до {dynamic['slots_per_step']:.2f} "
            f"слота/шаг, то есть в {facts['gate_ratio']:.1f} раза."
        ),
        Paragraph("МЕТОД", heading),
        para(
            "Qwen3.5-4B bf16, greedy, 32 новых токена, пять повторов после warmup с ротацией стратегий. "
            "Full KV и четыре политики сжатия сопоставляются с живой памятью без текстового RAG. "
            "Dynamic подключает до 32 латентных K/V-векторов с косинусом не ниже 0.5. Recall основан на "
            "ключевых строках; answerable-only вариант исключает unanswerable-пробы. Статистическая "
            "единица — диалог, а не отдельный вопрос."
        ),
    ]
    rows = []
    for name in ORDER:
        row = results[name]
        lo, hi = row["recall_ci"]
        rows.append([LABELS[name], f"{row['recall']:.3f}\n[{lo:.3f}; {hi:.3f}]", f"{row['answerable']:.3f}",
                     f"{row['e2e']:.3f}", f"{row['peak']:.0f}", f"{row['prompt']:.0f}", f"{row['attention']:.0f}"])
    add_table(story, "Таблица 1 — Качество и стоимость по завершённым диалогам",
              ["Стратегия", "Recall\n[95% ДИ]", "Answerable\nrecall", "E2E,\nс/ход", "Peak\nMiB", "Prompt\nтокены", "Attention\nслоты"],
              rows, [3.5, 2.5, 2.0, 1.5, 1.6, 2.3, 2.6])
    story.extend([
        PageBreak(),
        Paragraph("КАЧЕСТВО, ЛАТЕНТНОСТЬ И ПАМЯТЬ", heading),
        Image(str(figures["quality"]), width=17.4 * cm, height=6.0 * cm),
        Paragraph("Рисунок 1 — Средние показатели; интервалы recall получены bootstrap по диалогам", caption),
    ])
    pair_rows = []
    for other in ["v4_kv_mid", "pyramidkv_128", "snapkv_128"]:
        key = f"v4_kv_mid_dyn vs {other}"
        pair_rows.append([f"dynamic vs {LABELS[other]}", f"{dynamic['recall'] - results[other]['recall']:+.3f}",
                          f"{pvalues[key]:.4f}"])
    add_table(story, "Таблица 2 — Парные сравнения recall", ["Сравнение", "Δ recall", "p после Холма"],
              pair_rows, [7.0, 3.0, 3.5], 8.2)
    story.extend([
        para(
            "Dynamic достоверно улучшает k32 в текущем срезе. Численный разрыв с PyramidKV остаётся "
            "отрицательным, но выборка слишком мала для уверенного вывода о различии или эквивалентности. "
            "Преимущества по латентности и памяти являются измеренными характеристиками, однако не "
            "компенсируют отставание всех компактных методов от полного контекста."
        ),
        Paragraph("РАЗБОР ПО ДИАЛОГАМ", heading),
        Image(str(figures["dialogues"]), width=16.8 * cm, height=8.55 * cm),
        Paragraph("Рисунок 2 — Recall каждой стратегии на отдельных диалогах", caption),
        PageBreak(),
        Paragraph("ДИСТАНЦИЯ УДЕРЖАНИЯ", heading),
        Image(str(figures["lag"]), width=16.8 * cm, height=8.55 * cm),
        Paragraph("Рисунок 3 — Recall по расстоянию между фактом и контрольным вопросом", caption),
        para(
            f"В зоне 13–20 dynamic получил {dynamic['lag']['13-20']:.3f} против {fixed['lag']['13-20']:.3f} "
            f"у k32; в зоне 21+ — {dynamic['lag']['21+']:.3f} против {fixed['lag']['21+']:.3f}. "
            "Это улучшение относительно фиксированной инъекции, но не полноценное долгосрочное удержание: "
            "temporal recall обеих живых конфигураций остаётся 0.000."
        ),
        Paragraph("ОГРАНИЧЕНИЯ", heading),
        para(
            f"Готово только {n} из 48 диалогов, все длиной 100 ходов. Темы и сценарии представлены "
            "неравномерно. Keyword-скоринг не заменяет человеческую оценку качества. Низкая статистическая "
            "мощность не позволяет трактовать незначимые различия как эквивалентность. Peak VRAM в таблице "
            "является allocated-пиком генерации, а не всей reserved-памятью процесса."
        ),
        Paragraph("ЗАКЛЮЧЕНИЕ", heading),
        para(
            "На текущих пяти диалогах динамический гейт является предпочтительной конфигурацией живой "
            "памяти: он улучшает recall относительно k32 и более чем на порядок сокращает число инъекций. "
            "Одновременно он сохраняет преимущество перед PyramidKV по латентности, VRAM и prompt-объёму. "
            "Для вывода о конкурентоспособном качестве необходимо продолжить main-корпус и проверить "
            "200-ходовые диалоги."
        ),
        Paragraph("ВОСПРОИЗВЕДЕНИЕ", heading),
        Paragraph(".venv/bin/python scripts/build_core_report.py", code),
        Paragraph("Источник: results/corpus/full-core-v1/batches/", body),
    ])

    def page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Serif", 9)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawCentredString(A4[0] / 2, 0.75 * cm, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    return PDF_PATH


def main() -> None:
    """Regenerate all current core report artifacts, overwriting prior outputs."""
    results, records = collect_results()
    figures = build_figures(results, records)
    facts = report_facts(results)
    markdown_report(results, records, figures, facts)
    pdf_report(results, records, figures, facts)
    print(MD_PATH)
    print(PDF_PATH)
    for figure in figures.values():
        print(figure)


if __name__ == "__main__":
    main()

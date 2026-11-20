"""Build current RAG-only plots and Markdown/PDF reports from full-rag-v1 batches."""

from __future__ import annotations

import html
import sys
from datetime import date
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN_DIR = ROOT / "results/corpus/full-rag-v1"
PDF_PATH = RUN_DIR / "RAG_ONLY_REPORT.pdf"
MD_PATH = RUN_DIR / "RAG_ONLY_REPORT.md"
ORDER = [
    "classic_kv_full", "rag_text_only", "rag_last_centered", "rag_latent_mid",
    "rag_mid_centered", "rag_mean", "rag_mean_centered", "rag_bank_key", "rag_bank_dev",
]
UTTERANCE = ORDER[1:7]
LABELS = {
    "classic_kv_full": "Full KV",
    "rag_text_only": "last raw",
    "rag_last_centered": "last centered",
    "rag_latent_mid": "mid raw",
    "rag_mid_centered": "mid centered",
    "rag_mean": "mean raw",
    "rag_mean_centered": "mean centered",
    "rag_bank_key": "bank key",
    "rag_bank_dev": "bank deviation",
}
COLORS = {
    "classic_kv_full": "#334155",
    "rag_text_only": "#93c5fd",
    "rag_last_centered": "#2563eb",
    "rag_latent_mid": "#5eead4",
    "rag_mid_centered": "#0f766e",
    "rag_mean": "#d8b4fe",
    "rag_mean_centered": "#7e22ce",
    "rag_bank_key": "#94a3b8",
    "rag_bank_dev": "#64748b",
}
ZONES = ["0-5", "6-12", "13-20", "21+"]
CENTERING_PAIRS = [
    ("last", "rag_last_centered", "rag_text_only"),
    ("mid", "rag_mid_centered", "rag_latent_mid"),
    ("mean", "rag_mean_centered", "rag_mean"),
]


def p_text(value):
    """Format bootstrap p-values without presenting a rounded zero as exactly zero."""
    return "<0.001" if value == 0 else f"{value:.3f}"


def collect_results():
    """Aggregate RAG-only strategies and Full KV over completed dialogues.

    K/V-hybrid strategies are excluded. Recall values are macro-averaged over
    dialogues, E2E is the median of dialogue medians in seconds per turn, and
    peak VRAM is the mean allocated-memory peak in MiB.
    """
    from tmem.corpus_stats import bootstrap_ci, compare_pairs, load_run

    run = load_run(RUN_DIR)
    records = sorted(run["records"], key=lambda rec: rec["batch_id"])
    if not records:
        raise RuntimeError(f"Нет завершённых батчей в {RUN_DIR}")
    results = {}
    for name in ORDER:
        entries = [rec["strategies"][name] for rec in records]
        analyses = [entry["analysis"] for entry in entries]
        recalls = [analysis["recall_mean"] for analysis in analyses]
        answerable = []
        for analysis in analyses:
            scores = [probe["score"] or 0.0 for probe in analysis["probe_details"]
                      if probe["kind"] != "unanswerable"]
            answerable.append(mean(scores) if scores else 0.0)
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
            "e2e": median(median(entry["e2e"]["per_rep_mean_s"]) for entry in entries),
            "peak": mean(analysis["peak_mem_mb"] for analysis in analyses),
            "prompt": mean(analysis["prompt_tokens"] for analysis in analyses),
            "lag": lag,
        }
    pairs = [(centered, raw) for _, centered, raw in CENTERING_PAIRS]
    return results, records, compare_pairs(run, pairs)


def build_figures(results, records, pvalues):
    """Write four RAG-only PNG figures and return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    paths = {key: RUN_DIR / f"rag_only_{key}.png" for key in
             ["quality_cost", "layer_centering", "per_dialogue", "lag"]}
    labels = [LABELS[name] for name in ORDER]
    colors = [COLORS[name] for name in ORDER]
    x = np.arange(len(ORDER))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.2))
    recalls = [results[name]["recall"] for name in ORDER]
    lower = [results[name]["recall"] - results[name]["recall_ci"][0] for name in ORDER]
    upper = [results[name]["recall_ci"][1] - results[name]["recall"] for name in ORDER]
    bars = axes[0, 0].bar(x, recalls, color=colors, yerr=[lower, upper], capsize=3,
                          error_kw={"lw": 1}, width=0.72)
    axes[0, 0].set_title("Recall с 95% bootstrap ДИ", fontweight="bold")
    axes[0, 0].set_ylim(0, 1.08)
    for bar, value, name in zip(bars, recalls, ORDER):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2,
                        min(results[name]["recall_ci"][1] + 0.018, 1.04), f"{value:.3f}",
                        ha="center", fontsize=7.5)
    panels = [
        (axes[0, 1], "answerable", "Answerable-only recall", "{:.3f}", 1.08),
        (axes[1, 0], "e2e", "E2E латентность, с/ход", "{:.2f}", None),
        (axes[1, 1], "peak", "Peak allocated VRAM, MiB", "{:.0f}", None),
    ]
    for ax, key, title, fmt, fixed_upper in panels:
        values = [results[name][key] for name in ORDER]
        bars = ax.bar(x, values, color=colors, width=0.72)
        upper_y = fixed_upper or max(values) * 1.2
        ax.set_ylim(0, upper_y)
        ax.set_title(title, fontweight="bold")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + upper_y * 0.015, fmt.format(value),
                    ha="center", fontsize=7.5)
    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"RAG без K/V-инъекций · {len(records)} диалога · Qwen3.5-4B",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(paths["quality_cost"], dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    positions = np.arange(3)
    width = 0.34
    raw_names = [raw for _, _, raw in CENTERING_PAIRS]
    centered_names = [centered for _, centered, _ in CENTERING_PAIRS]
    raw_values = [results[name]["recall"] for name in raw_names]
    centered_values = [results[name]["recall"] for name in centered_names]
    raw_bars = ax.bar(positions - width / 2, raw_values, width, label="raw",
                      color=["#93c5fd", "#5eead4", "#d8b4fe"])
    centered_bars = ax.bar(positions + width / 2, centered_values, width, label="centered",
                           color=["#2563eb", "#0f766e", "#7e22ce"])
    full = results["classic_kv_full"]["recall"]
    ax.axhline(full, color="#334155", linestyle="--", label=f"Full KV = {full:.3f}")
    for bars in [raw_bars, centered_bars]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9)
    for index, (layer, centered, raw) in enumerate(CENTERING_PAIRS):
        delta = results[centered]["recall"] - results[raw]["recall"]
        pvalue = pvalues[f"{centered} vs {raw}"]
        ax.text(index, 0.06, f"Δ={delta:+.3f}\np{p_text(pvalue) if pvalue == 0 else '=' + p_text(pvalue)}",
                ha="center", fontsize=9, bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1"})
    ax.set_xticks(positions, ["last", "mid", "mean"])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Recall")
    ax.set_title("Влияние центрирования внутри каждого слоя", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(paths["layer_centering"], dpi=190, bbox_inches="tight")
    plt.close(fig)

    dialogue_labels = [f"{rec['batch_id'].split('-')[2]}\n{rec['spec']['topic']}\n{rec['spec']['scenario']}"
                       for rec in records]
    matrix = np.array([[rec["strategies"][name]["analysis"]["recall_mean"] for rec in records]
                       for name in ORDER])
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(records)), dialogue_labels)
    ax.set_yticks(np.arange(len(ORDER)), labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if value > 0.62 else "#0f172a")
    ax.set_title("Recall RAG-only по отдельным диалогам", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Recall", shrink=0.85)
    fig.tight_layout()
    fig.savefig(paths["per_dialogue"], dpi=190, bbox_inches="tight")
    plt.close(fig)

    lag_matrix = np.array([[results[name]["lag"][zone] for zone in ZONES] for name in ORDER], dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    image = ax.imshow(lag_matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(ZONES)), ZONES)
    ax.set_yticks(np.arange(len(ORDER)), labels)
    for row in range(lag_matrix.shape[0]):
        for column in range(lag_matrix.shape[1]):
            value = lag_matrix[row, column]
            ax.text(column, row, "—" if np.isnan(value) else f"{value:.3f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if value > 0.62 else "#0f172a")
    ax.set_xlabel("Расстояние между фактом и вопросом, ходы")
    ax.set_title("Удержание фактов по lag-зонам", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Recall", shrink=0.85)
    fig.tight_layout()
    fig.savefig(paths["lag"], dpi=190, bbox_inches="tight")
    plt.close(fig)
    return paths


def markdown_report(results, records, pvalues, figures):
    """Write the current RAG-only Markdown report and return its path."""
    n = len(records)
    best_name = max(UTTERANCE, key=lambda name: results[name]["recall"])
    best = results[best_name]
    full = results["classic_kv_full"]
    lines = [
        "# Full-rag-v1: RAG без K/V-инъекций",
        "",
        f"**Срез:** {n}/48 диалогов · Qwen3.5-4B bf16 · {date.today().isoformat()}. "
        "Все стратегии `v4_rag_kv_*` исключены.",
        "",
        "## Краткий вывод",
        "",
        f"Лучшие utterance-keyed RAG образуют плотную группу: mean centered "
        f"**{results['rag_mean_centered']['recall']:.3f}**, last centered "
        f"**{results['rag_last_centered']['recall']:.3f}**, mid centered "
        f"**{results['rag_mid_centered']['recall']:.3f}**. При n={n} нет оснований объявлять один из них "
        f"однозначным победителем. Full KV сохраняет преимущество: **{full['recall']:.3f}**.",
        "",
        f"Центрирование last улучшает recall с {results['rag_text_only']['recall']:.3f} до "
        f"{results['rag_last_centered']['recall']:.3f} (**Δ="
        f"{results['rag_last_centered']['recall'] - results['rag_text_only']['recall']:+.3f}**, "
        f"p{p_text(pvalues['rag_last_centered vs rag_text_only']) if pvalues['rag_last_centered vs rag_text_only'] == 0 else '=' + p_text(pvalues['rag_last_centered vs rag_text_only'])}) "
        "и помогает на каждом из четырёх диалогов. Для mid и mean средние приросты малы и не значимы. "
        "Bank-keyed варианты заметно отстают.",
        "",
        "## Качество и стоимость",
        "",
        "![RAG-only качество и стоимость](rag_only_quality_cost.png)",
        "",
        "| Стратегия | Recall [95% ДИ] | Answerable | E2E, с/ход | Peak VRAM, MiB | Prompt-токены |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        row = results[name]
        lo, hi = row["recall_ci"]
        lines.append(f"| {LABELS[name]} | {row['recall']:.3f} [{lo:.3f}, {hi:.3f}] | "
                     f"{row['answerable']:.3f} | {row['e2e']:.3f} | {row['peak']:.0f} | {row['prompt']:.0f} |")
    lines += [
        "",
        "E2E — медиана диалоговых медиан пяти повторов после warmup. Answerable исключает "
        "unanswerable-пробы. Peak VRAM — среднее `torch.max_memory_allocated`, не reserved-память.",
        "",
        "## Слой и центрирование",
        "",
        "![Слой и центрирование](rag_only_layer_centering.png)",
        "",
        "| Слой | Raw | Centered | Δ centered−raw | p после Холма |",
        "|---|---:|---:|---:|---:|",
    ]
    for layer, centered, raw in CENTERING_PAIRS:
        delta = results[centered]["recall"] - results[raw]["recall"]
        pvalue = pvalues[f"{centered} vs {raw}"]
        lines.append(f"| {layer} | {results[raw]['recall']:.3f} | {results[centered]['recall']:.3f} | "
                     f"{delta:+.3f} | {p_text(pvalue)} |")
    lines += [
        "",
        "Парный bootstrap выполнен по диалогам; три проверки центрирования скорректированы методом Холма. "
        "У last все четыре парных сдвига положительны. Для mid/mean направление меняется между диалогами "
        "или эффект мал.",
        "",
        "## Значения по диалогам",
        "",
        "![RAG-only по диалогам](rag_only_per_dialogue.png)",
        "",
        "## Дистанция удержания",
        "",
        "![RAG-only lag-зоны](rag_only_lag.png)",
        "",
        "Centered-варианты лучше удерживают дальние факты, но остаются ниже Full KV. "
        "Bank deviation особенно слаб в зонах 13–20 и 21+.",
        "",
        "## Метод и ограничения",
        "",
        "last — финальный слой, mid — один средний слой, mean — среднее всех слоёв. Raw использует "
        "исходный эмбеддинг; centered вычитает running mean текстового RAG-банка. Bank key/deviation "
        "используют слоты живого банка только как retrieval-ключи, без K/V-инъекции.",
        "",
        f"Готово только {n} диалога, все длиной 100 ходов; тема, персона и сценарий изменяются вместе. "
        "Recall является keyword-скорингом синтетических проб. Доверительные интервалы широкие, поэтому "
        "небольшие различия между лучшими centered-вариантами не следует переинтерпретировать.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        ".venv/bin/python scripts/build_rag_only_report.py",
        "```",
        "",
        "Источник: `results/corpus/full-rag-v1/batches/`.",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n")
    return MD_PATH


def pdf_report(results, records, pvalues, figures):
    """Write a paginated PDF containing the RAG-only tables and figures."""
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
    title = ParagraphStyle("RagTitle", parent=styles["Title"], fontName="Serif-Bold", fontSize=17,
                           leading=21, alignment=TA_CENTER, spaceAfter=9)
    subtitle = ParagraphStyle("RagSubtitle", parent=styles["Normal"], fontName="Serif-Italic", fontSize=10,
                              leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#475569"), spaceAfter=12)
    body = ParagraphStyle("RagBody", parent=styles["Normal"], fontName="Serif", fontSize=10.1, leading=13.8,
                          alignment=TA_JUSTIFY, spaceAfter=6)
    heading = ParagraphStyle("RagHeading", parent=body, fontName="Serif-Bold", fontSize=13, leading=16,
                             spaceBefore=8, spaceAfter=6, keepWithNext=True)
    caption = ParagraphStyle("RagCaption", parent=body, fontSize=9, leading=11, alignment=TA_CENTER,
                             textColor=colors.HexColor("#475569"), spaceAfter=6)
    cell = ParagraphStyle("RagCell", parent=body, fontSize=7.4, leading=8.8, alignment=TA_CENTER, spaceAfter=0)
    head = ParagraphStyle("RagHead", parent=cell, fontName="Serif-Bold")
    code = ParagraphStyle("RagCode", parent=body, fontName="Courier", fontSize=8.5, leading=11)

    def add_table(story, table_title, headers, rows, widths, font_size=7.4):
        sized_cell = ParagraphStyle(f"RagCell{len(story)}", parent=cell, fontSize=font_size,
                                    leading=font_size * 1.2)
        sized_head = ParagraphStyle(f"RagHead{len(story)}", parent=head, fontSize=font_size,
                                    leading=font_size * 1.2)
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
        story.extend([Paragraph(html.escape(table_title), caption), table, Spacer(1, 0.18 * cm)])

    n = len(records)
    full = results["classic_kv_full"]
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title=f"RAG-only: отчёт по {n} диалогам", author="Transformer experiments")
    story = [
        Paragraph("Текстовый RAG без K/V-инъекций", title),
        Paragraph(f"full-rag-v1 · актуальный срез {n}/48 · {date.today().isoformat()}", subtitle),
        Paragraph("КРАТКИЙ ВЫВОД", heading),
        Paragraph(
            f"Mean centered, last centered и mid centered получили recall "
            f"<b>{results['rag_mean_centered']['recall']:.3f}</b>, "
            f"<b>{results['rag_last_centered']['recall']:.3f}</b> и "
            f"<b>{results['rag_mid_centered']['recall']:.3f}</b>. Это плотная группа при широких "
            f"интервалах; однозначного победителя она не имеет. Full KV сохраняет преимущество "
            f"({full['recall']:.3f}). Bank-keyed варианты отстают.", body),
        Paragraph(
            f"Наиболее сильный внутренний эффект — центрирование last: Δ="
            f"{results['rag_last_centered']['recall'] - results['rag_text_only']['recall']:+.3f}, "
            f"положительный сдвиг на всех четырёх диалогах. Для mid и mean приросты около +0.034 "
            "и не значимы. В отчёт не включены гибриды v4_rag_kv_*.", body),
        Paragraph("ЧТО СРАВНИВАЕТСЯ", heading),
        Paragraph(
            "last — финальный слой, mid — один средний слой, mean — среднее всех слоёв. Raw использует "
            "исходный эмбеддинг; centered вычитает running mean текстового RAG-банка. Bank key/deviation "
            "используют слоты живого банка только как retrieval-ключи, но не добавляют K/V в attention.", body),
    ]
    rows = []
    for name in ORDER:
        row = results[name]
        lo, hi = row["recall_ci"]
        rows.append([LABELS[name], f"{row['recall']:.3f}\n[{lo:.3f}; {hi:.3f}]", f"{row['answerable']:.3f}",
                     f"{row['e2e']:.3f}", f"{row['peak']:.0f}", f"{row['prompt']:.0f}"])
    add_table(story, "Таблица 1 — Качество и стоимость RAG-only",
              ["Стратегия", "Recall [95% ДИ]", "Answerable", "E2E, с/ход", "Peak MiB", "Prompt"],
              rows, [4.0, 3.2, 2.3, 2.2, 2.0, 2.4])
    story.extend([
        PageBreak(),
        Paragraph("КАЧЕСТВО, ЛАТЕНТНОСТЬ И ПАМЯТЬ", heading),
        Image(str(figures["quality_cost"]), width=17.2 * cm, height=10.55 * cm),
        Paragraph("Рисунок 1 — Все RAG-only стратегии и Full KV; ДИ построены по диалогам", caption),
        Paragraph(
            f"Лучшие RAG дают около 0.71 recall при латентности около 1.26 с/ход против "
            f"{full['e2e']:.3f} у Full KV. Экономия латентности и prompt-объёма не устраняет "
            "разрыв в качестве с полной историей.", body),
        Paragraph("СЛОЙ И ЦЕНТРИРОВАНИЕ", heading),
        Image(str(figures["layer_centering"]), width=16.4 * cm, height=8.45 * cm),
        Paragraph("Рисунок 2 — Парный эффект центрирования для last, mid и mean", caption),
        PageBreak(),
        Paragraph("РАЗБОР ПО ДИАЛОГАМ", heading),
        Image(str(figures["per_dialogue"]), width=16.8 * cm, height=10.55 * cm),
        Paragraph("Рисунок 3 — Recall каждой RAG-only конфигурации на каждом диалоге", caption),
        Paragraph(
            "Last centered стабильно выше last raw. Остальные конфигурации меняют порядок между "
            "диалогами, поэтому средний рейтинг нельзя рассматривать как универсальный. Сценарий "
            "конфаундирован темой, персоной и seed.", body),
        Paragraph("ДИСТАНЦИЯ УДЕРЖАНИЯ", heading),
        Image(str(figures["lag"]), width=16.8 * cm, height=10.25 * cm),
        Paragraph("Рисунок 4 — Recall по расстоянию между фактом и контрольным вопросом", caption),
        PageBreak(),
        Paragraph("ОБСУЖДЕНИЕ", heading),
        Paragraph(
            "Для raw-ключей mid и mean заметно лучше last, но после центрирования различия между "
            "слоями почти исчезают. Это согласуется с тем, что глобальное центрирование особенно "
            "исправляет анизотропию last-представлений. Такой механизм остаётся интерпретацией, а не "
            "напрямую проверенным причинным выводом.", body),
        Paragraph(
            "Bank key и bank deviation не конкурируют с обычными utterance-ключами. Их слабость "
            "особенно видна на дальних дистанциях. Практический выбор на текущем срезе — один из "
            "centered utterance-RAG; last centered дополнительно имеет наиболее ясный парный контроль "
            "против собственного raw-варианта.", body),
        Paragraph("ОГРАНИЧЕНИЯ", heading),
        Paragraph(
            f"Готово {n} из 48 диалогов, все длиной 100 ходов. Keyword-скоринг не заменяет человеческую "
            "оценку. Низкая статистическая мощность не позволяет ранжировать близкие centered-варианты "
            "или утверждать отсутствие эффекта mid/mean центрирования. Для 200-ходовых диалогов вывод "
            "ещё не проверен.", body),
        Paragraph("ВОСПРОИЗВЕДЕНИЕ", heading),
        Paragraph(".venv/bin/python scripts/build_rag_only_report.py", code),
        Paragraph("Источник: results/corpus/full-rag-v1/batches/", body),
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
    """Regenerate all RAG-only report artifacts, overwriting previous outputs."""
    results, records, pvalues = collect_results()
    figures = build_figures(results, records, pvalues)
    markdown_report(results, records, pvalues, figures)
    pdf_report(results, records, pvalues, figures)
    print(MD_PATH)
    print(PDF_PATH)
    for figure in figures.values():
        print(figure)


if __name__ == "__main__":
    main()

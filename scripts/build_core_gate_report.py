"""Build a shareable PDF report from the first full-core-v1 corpus batch."""

from __future__ import annotations

import html
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results/corpus/full-core-v1"
BATCH_ID = "main-L100-00-work-terse-baseline"
BATCH_DIR = RUN_DIR / "batches" / BATCH_ID
PDF_PATH = RUN_DIR / "CORE_GATE_REPORT.pdf"
FIGURE_PATH = RUN_DIR / "core_gate_comparison.png"
ORDER = [
    "classic_kv_full", "sparse_sliding_128", "sparse_topk_128", "snapkv_128",
    "pyramidkv_128", "v4_kv_mid", "v4_kv_mid_dyn",
]
LABELS = {
    "classic_kv_full": "Full KV",
    "sparse_sliding_128": "Sliding 128",
    "sparse_topk_128": "TopK 128",
    "snapkv_128": "SnapKV 128",
    "pyramidkv_128": "PyramidKV 128",
    "v4_kv_mid": "Живая память k32",
    "v4_kv_mid_dyn": "Живая память dynamic",
}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    results = {}
    for name in ORDER:
        payload = json.loads((BATCH_DIR / f"{name}.json").read_text())
        details = payload["analysis"]["probe_details"]
        answerable = [p["score"] or 0.0 for p in details if p["kind"] != "unanswerable"]
        results[name] = {
            "recall": payload["analysis"]["recall_mean"],
            "answerable": sum(answerable) / len(answerable),
            "e2e": median(payload["e2e"]["per_rep_mean_s"]),
            "peak": payload["analysis"]["peak_mem_mb"],
            "prompt": payload["analysis"]["prompt_tokens"],
            "attention": payload["analysis"]["attention_flops_proxy"],
            "kv_slots": payload["analysis"]["kv_injected_slots"],
            "kv_steps": payload["analysis"]["kv_steps"],
        }

    dynamic = results["v4_kv_mid_dyn"]
    fixed = results["v4_kv_mid"]
    pyramid = results["pyramidkv_128"]
    snap = results["snapkv_128"]
    full = results["classic_kv_full"]
    speed_gain = (1 - dynamic["e2e"] / pyramid["e2e"]) * 100
    vram_gain = (1 - dynamic["peak"] / pyramid["peak"]) * 100
    prompt_factor = pyramid["prompt"] / dynamic["prompt"]
    attention_factor = dynamic["attention"] / pyramid["attention"]
    gate_ratio = fixed["kv_slots"] / fixed["kv_steps"] / (dynamic["kv_slots"] / dynamic["kv_steps"])

    palette = {
        "classic_kv_full": "#334155", "sparse_sliding_128": "#94a3b8",
        "sparse_topk_128": "#64748b", "snapkv_128": "#3b82f6",
        "pyramidkv_128": "#8b5cf6", "v4_kv_mid": "#0d9488",
        "v4_kv_mid_dyn": "#059669",
    }
    labels = [LABELS[name] for name in ORDER]
    values = [results[name] for name in ORDER]
    x = range(len(ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    panels = [
        ("recall", "Recall", "{:.3f}", 1.05),
        ("e2e", "E2E латентность, с/ход", "{:.2f}", 2.35),
        ("peak", "Peak VRAM, MiB", "{:.0f}", 12500),
    ]
    for ax, (key, title, fmt, upper) in zip(axes, panels):
        series = [row[key] for row in values]
        bars = ax.bar(x, series, color=[palette[name] for name in ORDER], width=0.75)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(list(x), labels, rotation=32, ha="right", fontsize=8)
        ax.set_ylim(0, upper)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, series):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + upper * 0.015,
                    fmt.format(value), ha="center", va="bottom", fontsize=8)
    fig.suptitle("full-core-v1 · первый main-диалог · n=1", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=190, bbox_inches="tight")
    plt.close(fig)

    fonts = Path(matplotlib.get_data_path()) / "fonts/ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(fonts / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], fontName="DejaVu-Bold",
                           fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"))
    subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName="DejaVu",
                              fontSize=9, leading=12, textColor=colors.HexColor("#475569"),
                              spaceAfter=10)
    body = ParagraphStyle("ReportBody", parent=styles["Normal"], fontName="DejaVu",
                          fontSize=9.2, leading=13, spaceAfter=6)
    heading = ParagraphStyle("ReportHeading", parent=styles["Heading2"], fontName="DejaVu-Bold",
                             fontSize=12, leading=15, spaceBefore=7, spaceAfter=6,
                             textColor=colors.HexColor("#0f172a"), keepWithNext=True)
    small = ParagraphStyle("TableCell", parent=body, fontSize=7.5, leading=9, spaceAfter=0)
    small_head = ParagraphStyle("TableHead", parent=small, fontName="DejaVu-Bold")

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=landscape(A4),
                            leftMargin=1.15 * cm, rightMargin=1.15 * cm,
                            topMargin=1.0 * cm, bottomMargin=1.0 * cm,
                            title="Живая память на диалогах: full-core-v1",
                            author="Transformer experiments",
                            subject="Первый контрольный батч full-core-v1")
    story = [
        Paragraph("Живая память на диалогах: сравнение с KV-кэшем", title),
        Paragraph("full-core-v1 · main-L100-00-work-terse-baseline · Qwen3.5-4B bf16 · 17.11.2026", subtitle),
        Paragraph("Основной результат", heading),
        Paragraph(
            f"Динамическая живая память получила recall <b>{dynamic['recall']:.3f}</b> против "
            f"{pyramid['recall']:.3f} у PyramidKV: на {speed_gain:.1f}% ниже end-to-end латентность, "
            f"на {vram_gain:.1f}% меньше peak VRAM и в {prompt_factor:.2f} раза меньше prompt-токенов. "
            f"SnapKV сохранил небольшое преимущество по качеству ({snap['recall']:.3f}), а полный контекст "
            f"остался лучшим ({full['recall']:.3f}). Это направленческий результат одного диалога, "
            "а не статистически подтверждённое преимущество.", body,
        ),
        Paragraph("Качество и стоимость", heading),
    ]
    headers = ["Стратегия", "Recall", "Без\nunanswerable", "E2E, с/ход", "Peak VRAM,\nMiB", "Prompt\nтокены", "Attention\nслоты"]
    table_rows = []
    for name in ORDER:
        row = results[name]
        table_rows.append([
            LABELS[name], f"{row['recall']:.3f}", f"{row['answerable']:.3f}",
            f"{row['e2e']:.3f}", f"{row['peak']:.0f}", f"{row['prompt']:,}".replace(",", " "),
            f"{row['attention']:,}".replace(",", " "),
        ])
    data = [[Paragraph(html.escape(cell), small_head) for cell in headers]]
    data += [[Paragraph(html.escape(cell), small) for cell in row] for row in table_rows]
    table = Table(data, colWidths=[5.0 * cm, 1.8 * cm, 2.3 * cm, 2.2 * cm, 2.5 * cm, 3.0 * cm, 3.2 * cm],
                  repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#94a3b8")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 6), (-1, 6), colors.HexColor("#ccfbf1")),
        ("BACKGROUND", (0, 7), (-1, 7), colors.HexColor("#d1fae5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [table, Spacer(1, 0.2 * cm),
              Image(str(FIGURE_PATH), width=23.7 * cm, height=7.8 * cm), PageBreak()]

    interpretation = [
        f"<b>Dynamic против PyramidKV:</b> +0.136 recall, −{speed_gain:.1f}% e2e, −{vram_gain:.1f}% peak VRAM. "
        "Диапазоны пяти замеров латентности не пересекаются.",
        f"<b>SnapKV всё ещё выше по качеству:</b> {snap['recall']:.3f} против {dynamic['recall']:.3f}. "
        "Поэтому результат не означает лидерства среди всех компактных стратегий.",
        f"<b>Динамический гейт эффективнее фиксированного k32:</b> +0.045 recall; в среднем "
        f"{dynamic['kv_slots'] / dynamic['kv_steps']:.2f} слота/шаг против "
        f"{fixed['kv_slots'] / fixed['kv_steps']:.2f} — в {gate_ratio:.1f} раза меньше.",
        f"<b>Компромисс:</b> attention-прокси dynamic в {attention_factor:.2f} раза выше, чем у PyramidKV "
        f"({dynamic['attention']:,} против {pyramid['attention']:,} слотов). Экономия prompt-токенов не означает "
        "такую же экономию работы attention.",
        "<b>Слабое место:</b> обе живые конфигурации получили 0.000 в lag-зоне 13–20. "
        "У dynamic recall зоны 21+ равен 0.278; преимущество получено в основном на коротких дистанциях и paraphrase-пробах.",
        "<b>Full KV остаётся эталоном качества:</b> 0.955 overall и 0.950 после исключения unanswerable-проб.",
    ]
    story.append(Paragraph("Интерпретация", heading))
    story += [Paragraph(f"• {item}", body) for item in interpretation]
    story.append(Paragraph("Протокол", heading))
    protocol = [
        "Один диалог main: работа/IT, лаконичная персона, baseline; 100 ходов, 22 пробы, seed=5000.",
        "Qwen3.5-4B bf16 на AMD RX 7900 XT; greedy; до 32 новых токенов; KV-бюджет контролей 128.",
        "Один warmup и пять чередующихся повторов каждой стратегии. E2E включает prefill, decoding и retrieval/embedding.",
        "Recall — keyword-скоринг. Колонка «Без unanswerable» исключает две грубые unanswerable-пробы.",
        "Peak VRAM — torch.max_memory_allocated внутри генерации; attention-слоты — прокси, не прямой замер FLOPs.",
    ]
    story += [Paragraph(f"• {html.escape(item)}", body) for item in protocol]
    story.append(Paragraph("Ограничения и следующий шаг", heading))
    story.append(Paragraph(
        "n=1: доверительный интервал по диалогам и парная значимость пока недоступны. В baseline нет temporal-проб. "
        "Для подтверждения нужны остальные темы, персоны и сценарии main. Следующие независимые контрольные батчи — "
        "self (настройки живой памяти) и rag (текстовые каналы).", body,
    ))
    story.append(Paragraph("Воспроизведение", heading))
    story += [
        Paragraph("scripts/run_dialogue_fullscale.sh core gate", body),
        Paragraph(".venv/bin/python scripts/build_core_gate_report.py", body),
        Paragraph("Данные: results/corpus/full-core-v1/batches/main-L100-00-work-terse-baseline/", subtitle),
    ]

    def page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawRightString(landscape(A4)[0] - 1.15 * cm, 0.45 * cm, f"стр. {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(PDF_PATH)


if __name__ == "__main__":
    main()

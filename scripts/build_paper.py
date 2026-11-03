"""Render the scientific article as PAPER.md and PAPER.pdf."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_text as PT
import report_data as RD

CONFIG_HEADERS = ["Система", "Контекст", "Память и бюджет"]
CONFIG_ROWS = [
    ["Чистый Qwen", "Вся история", "Классический неограниченный KV-кэш"],
    ["StreamingLLM", "Вся история", "128 слотов: 4 sink + 124 recent"],
    ["H2O", "Вся история", "128: 4 sink + 64 recent + до 60 heavy hitters"],
    ["SnapKV", "Вся история", "128; observation=64; kernel=7"],
    ["PyramidKV", "Вся история", "Среднее 128; observation=8; β=20"],
    ["Текстовый RAG-контроль", "Буфер 5 ходов + top-4", "Только текст; сырые эмбеддинги; классический KV"],
    ["Двухканальная память", "Буфер 5 ходов + top-4", "Текст + центрирование; латентный retrieval: до 32 K/V в 4 слоях"],
]

REPRO_COMMAND = (
    "HF_HUB_OFFLINE=1 .venv/bin/python -m tmem.cli --model unused --battery100 "
    "--models qwen35-4b --device cuda --dtype bfloat16 --max-new-tokens 32 "
    "--strategies classic_kv_full sparse_sliding_128 sparse_topk_128 snapkv_128 "
    "pyramidkv_128 rag_text_only v4_rag_kv --out results/qwen35_comparison"
)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def build_md(q: dict):
    lines = [
        f"# {PT.TITLE}", "", f"_{PT.AUTHORS}_", "", "## Аннотация", "", PT.abstract(q), "",
        f"**Ключевые слова:** {PT.KEYWORDS}", "",
    ]
    for heading, paragraphs in PT.sections(q):
        lines += [f"## {heading}", ""]
        lines += [paragraph + "\n" for paragraph in paragraphs]
        if heading.startswith("3."):
            lines += ["![Архитектура двухканальной памяти](figs/architecture.png)",
                      "*Рисунок 1 — Архитектура предложенной системы.*", ""]
        if heading.startswith("4."):
            lines += ["**Таблица 1 — Параметры сравниваемых систем.**", ""]
            lines += markdown_table(CONFIG_HEADERS, CONFIG_ROWS) + [""]
        if heading.startswith("5."):
            for caption, headers, rows in (
                ("Таблица 2 — Качество ответов на контрольные вопросы.", RD.QUALITY_HEADERS, RD.quality_rows(q)),
                ("Таблица 3 — Вычислительная стоимость и наблюденная латентность.", RD.COST_HEADERS, RD.cost_rows(q)),
            ):
                lines += [f"**{caption}**", ""] + markdown_table(headers, rows) + [""]
            lines += [
                "![Основное сравнение](figs/comparison.png)",
                "*Рисунок 2 — Качество, attention-прокси и латентность на общей батарее.*", "",
                "![Парето](figs/pareto.png)",
                "*Рисунок 3 — Компромисс между recall и decode-side attention-прокси.*", "",
            ]
    lines += ["## Список литературы", ""] + [f"{i}. {ref}" for i, ref in enumerate(PT.REFERENCES, 1)]
    lines += [
        "", "## Доступность данных и воспроизводимость", "",
        "Первичные результаты: `results/qwen35_comparison/qwen35-4b_battery100.json`. "
        "Файл содержит параметры протокола, агрегированные метрики и ответ на каждый из 24 вопросов.", "",
        "```bash", REPRO_COMMAND,
        ".venv/bin/python scripts/build_report.py", ".venv/bin/python scripts/build_paper.py", "```",
    ]
    (RD.OUT / "PAPER.md").write_text("\n".join(lines) + "\n")


def build_pdf(q: dict):
    import matplotlib
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(fonts / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Italic", str(fonts / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Italic")
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="DejaVu", fontSize=8.8,
                          leading=12.1, alignment=4, spaceAfter=6)
    heading = ParagraphStyle("heading", parent=styles["Heading2"], fontName="DejaVu-Bold",
                             fontSize=12.5, leading=16, spaceBefore=9, spaceAfter=6)
    small = ParagraphStyle("small", parent=body, fontSize=7.0, leading=8.8, alignment=0)
    caption = ParagraphStyle("caption", parent=small, fontName="DejaVu-Italic", alignment=1, spaceAfter=7)
    title = ParagraphStyle("title", parent=styles["Title"], fontName="DejaVu-Bold", fontSize=16, leading=21)
    authors = ParagraphStyle("authors", parent=body, fontName="DejaVu-Italic", alignment=1)
    doc = SimpleDocTemplate(
        str(RD.OUT / "PAPER.pdf"), pagesize=landscape(A4),
        leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.4 * cm,
        title=PT.TITLE, author=PT.AUTHORS, subject="Диалоговая память и сжатие KV-кэша",
    )

    def on_page(canvas, document):
        canvas.saveState()
        canvas.setFont("DejaVu", 7)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(1.5 * cm, .65 * cm, "Двухканальная смысловая память · Qwen3.5-4B")
        canvas.drawRightString(landscape(A4)[0] - 1.5 * cm, .65 * cm, str(document.page))
        canvas.restoreState()

    def paragraph(text):
        match = re.match(r"^(\d+\.\d+\.\s+[^.]+\.)\s+(.*)$", text, re.DOTALL)
        if match:
            text = f"<b>{match.group(1)}</b> {match.group(2)}"
        return Paragraph(text, body)

    def add_table(story, caption_text, headers, rows, first_width=5.0):
        story.append(Paragraph(caption_text, heading))
        data = [[Paragraph(h, small) for h in headers]] + [[Paragraph(str(c), small) for c in row] for row in rows]
        remaining = 26.7 - first_width
        widths = [first_width * cm] + [remaining / (len(headers) - 1) * cm] * (len(headers) - 1)
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9edf3")),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#8993a4")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story += [table, Spacer(1, .25 * cm)]

    story = [
        Paragraph(PT.TITLE, title), Paragraph(PT.AUTHORS, authors), Spacer(1, .2 * cm),
        Paragraph("Аннотация", heading), Paragraph(PT.abstract(q), body),
        Paragraph(f"<b>Ключевые слова:</b> {PT.KEYWORDS}", body),
    ]
    for section, paragraphs in PT.sections(q):
        story.append(Paragraph(section, heading))
        story += [paragraph(text) for text in paragraphs]
        if section.startswith("3."):
            story += [Image(str(RD.FIGS / "architecture.png"), width=23.6 * cm, height=12.6 * cm),
                      Paragraph("Рисунок 1 — Архитектура предложенной двухканальной памяти.", caption)]
        if section.startswith("4."):
            add_table(story, "Таблица 1 — Параметры сравниваемых систем", CONFIG_HEADERS, CONFIG_ROWS, 4.3)
        if section.startswith("5."):
            add_table(story, "Таблица 2 — Качество ответов", RD.QUALITY_HEADERS, RD.quality_rows(q), 5.0)
            add_table(story, "Таблица 3 — Стоимость и наблюденная латентность", RD.COST_HEADERS, RD.cost_rows(q), 4.0)
            story += [Image(str(RD.FIGS / "comparison.png"), width=24.0 * cm, height=14.8 * cm),
                      Paragraph("Рисунок 2 — Качество, attention-прокси и латентность на общей батарее.", caption),
                      Image(str(RD.FIGS / "pareto.png"), width=21.5 * cm, height=12.7 * cm),
                      Paragraph("Рисунок 3 — Компромисс между recall и decode-side attention-прокси.", caption)]
    story.append(Paragraph("Список литературы", heading))
    story += [Paragraph(f"[{i}] {ref}", small) for i, ref in enumerate(PT.REFERENCES, 1)]
    story.append(Paragraph("Доступность данных и воспроизводимость", heading))
    story.append(Paragraph(
        "Первичные результаты и ответы на каждый вопрос: "
        "results/qwen35_comparison/qwen35-4b_battery100.json. Команда запуска батареи и сборки документов "
        "приведена в электронной версии PAPER.md.", small))
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main():
    q = RD.q35_all()
    missing = set(RD.ORDER) - set(q)
    if missing:
        raise SystemExit(f"не хватает результатов: {sorted(missing)}")
    import paper_figures
    paper_figures.make_all()
    build_md(q)
    build_pdf(q)
    print("paper:", RD.OUT / "PAPER.pdf")


if __name__ == "__main__":
    main()

"""Render the paper (PAPER.md + PAPER.pdf) from paper_text + data tables."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_data as RD
import paper_text as PT
from build_report import rows_for_pdf, small_model_rows, HEADERS, GLOSS

OUT = RD.OUT
FIGS = RD.FIGS

FIG_PLAN = [
    ("architecture.png", "Рис. 1. Архитектура двухслойной памяти"),
    ("paper_main.png", "Рис. 2. Базовая модель vs разрежённое внимание vs наша система (4 оси)"),
    ("evolution.png", "Рис. 3. Эволюция качества и стоимости по ступеням"),
    ("pareto.png", "Рис. 4. Парето «точность фактов — attention-стоимость»"),
    ("smallmodels.png", "Рис. 5. Базовые политики на малых моделях"),
    ("training.png", "Рис. 6. Обучение читателя памяти: негативный результат"),
]


def build_md(q):
    lines = [f"# {PT.TITLE}", "", f"_{PT.AUTHORS}_", "", "## Аннотация", "", PT.ABSTRACT, ""]
    for heading, paras in PT.SECTIONS:
        lines += [f"## {heading}", ""]
        lines += [p + "\n" for p in paras]
        if heading.startswith("5."):
            lines += ["**Таблица 1. Основная батарея (Qwen3.5-4B, 107 ходов / 24 вопроса)**", "",
                      "| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
            lines += ["| " + " | ".join(r) + " |" for r in rows_for_pdf(q)]
            lines += ["", "**Таблица 2. Малые модели (базовое сравнение политик)**", "",
                      "| Модель: политика | Факты | Prompt токены | Attn слоты | Лат. с/ход |", "|---|---|---|---|---|"]
            lines += ["| " + " | ".join(r) + " |" for r in small_model_rows()]
            lines += [""]
        if heading.startswith("4."):
            lines += ["**Метрики**", "", "| Метрика | Определение |", "|---|---|"]
            lines += [f"| {a} | {b} |" for a, b in GLOSS] + [""]
    lines += ["## Список литературы", ""] + [f"{i+1}. {r}" for i, r in enumerate(PT.REFERENCES)]
    lines += ["", "## Воспроизведение", "", "```bash",
              ".venv/bin/python -m tmem.cli --model unused --battery100 --models qwen35-4b --device cuda --dtype bfloat16",
              ".venv/bin/python -m tmem.cli --model unused --stress --models gpt2 t5-base qwen2-1.5b --device cuda",
              ".venv/bin/python -m tmem.trainreader --steps 600",
              ".venv/bin/python scripts/build_paper.py", "```"]
    (OUT / "PAPER.md").write_text("\n".join(lines))


def build_pdf(q):
    import matplotlib
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                    Image, PageBreak)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(fonts / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")

    st = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=st["Normal"], fontName="DejaVu", fontSize=9, leading=12.5,
                          alignment=4, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=st["Heading1"], fontName="DejaVu-Bold", fontSize=13, leading=16,
                        spaceBefore=10, spaceAfter=6)
    title = ParagraphStyle("t", parent=st["Title"], fontName="DejaVu-Bold", fontSize=15, leading=20)
    small = ParagraphStyle("s", parent=body, fontSize=7.4, leading=9.4, alignment=0)

    doc = SimpleDocTemplate(str(OUT / "PAPER.pdf"), pagesize=landscape(A4),
                            leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                            title=PT.TITLE)
    story = [Paragraph(PT.TITLE, title), Paragraph(PT.AUTHORS, st["Normal"]), Spacer(1, 0.3 * cm),
             Paragraph("Аннотация", h1), Paragraph(PT.ABSTRACT, body), Spacer(1, 0.2 * cm),
             Paragraph("<b>Ключевые числа:</b> факты 83 % эталона при attention ×5 дешевле · тон 1.07× эталона · "
                       "латентность −28 % · разрежённый KV-кэш той же задачи теряет 83–100 % фактов · "
                       "пост-hoc обучение читателя — негативно (n=24×3 попытки).", body)]

    fig_page = {"5. Результаты": ("paper_main.png", 0), "3. Идея и архитектура": ("architecture.png", 0)}
    inserted_tables = False
    for heading, paras in PT.SECTIONS:
        story.append(Paragraph(heading, h1))
        for p in paras:
            story.append(Paragraph(p, body))
        if heading.startswith("3."):
            story.append(Image(str(FIGS / "architecture.png"), width=24.5 * cm, height=15.8 * cm))
            story.append(Paragraph("Рис. 1. Архитектура двухслойной памяти.", small))
        if heading.startswith("4."):
            g = [[Paragraph("Метрика", small), Paragraph("Определение", small)]]
            g += [[Paragraph(a, small), Paragraph(b, small)] for a, b in GLOSS]
            t = Table(g, colWidths=[7 * cm, 21 * cm])
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
            story.append(t)
        if heading.startswith("5.") and not inserted_tables:
            inserted_tables = True
            story.append(Paragraph("Таблица 1. Основная батарея (Qwen3.5-4B, 107 ходов / 24 вопроса)", h1))
            rows = rows_for_pdf(q)
            data = [[Paragraph(h, small) for h in HEADERS]] + [[Paragraph(c, small) for c in r] for r in rows]
            t = Table(data, colWidths=[9.6 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 2.3 * cm, 2.3 * cm,
                                       1.9 * cm, 2.1 * cm], repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                   ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(.96, .97, 1)]),
                                   ("BACKGROUND", (0, len(rows)), (-1, len(rows)), colors.Color(.9, .95, 1))]))
            story.append(t)
            story.append(PageBreak())
            story.append(Image(str(FIGS / "paper_main.png"), width=24.5 * cm, height=16.6 * cm))
            story.append(Paragraph("Рис. 2. Базовая модель (полный контекст + классический KV) vs разрежённое "
                                   "внимание (скользящий/H2O KV) vs наша система — по четырём осям.", small))
            story.append(PageBreak())
            story.append(Image(str(FIGS / "evolution.png"), width=22.5 * cm, height=15 * cm))
            story.append(Paragraph("Рис. 3. Эволюция системы по ступеням 1→4.", small))
            story.append(Spacer(1, 0.2 * cm))
            story.append(Image(str(FIGS / "pareto.png"), width=21.5 * cm, height=14 * cm))
            story.append(Paragraph("Рис. 4. Парето «точность фактов — стоимость внимания».", small))
        if heading.startswith("6."):
            story.append(PageBreak())
            story.append(Image(str(FIGS / "smallmodels.png"), width=20.5 * cm, height=11.6 * cm))
            story.append(Paragraph("Рис. 5. Базовые политики на малых моделях.", small))
            story.append(Spacer(1, 0.2 * cm))
            hdr = ["Модель: политика", "Факты", "Prompt токены", "Attn слоты", "Лат. с/ход"]
            data = [[Paragraph(h, small) for h in hdr]] + [[Paragraph(c, small) for c in r]
                                                           for r in small_model_rows()]
            t = Table(data, colWidths=[9 * cm, 2 * cm, 3 * cm, 3 * cm, 2.6 * cm], repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
            story.append(t)
            story.append(Spacer(1, 0.25 * cm))
            story.append(Image(str(FIGS / "training.png"), width=20.5 * cm, height=11.7 * cm))
            story.append(Paragraph("Рис. 6. Обучение читателя памяти (негативный результат при исправном "
                                   "позитивном контроле).", small))
    story.append(Paragraph("Список литературы", h1))
    for i, r in enumerate(PT.REFERENCES, 1):
        story.append(Paragraph(f"{i}. {r}", small))
    doc.build(story)


def main():
    q = RD.q35_all()
    if not q:
        raise SystemExit("нет данных")
    build_md(q)
    build_pdf(q)
    print("paper:", OUT / "PAPER.pdf", "и", OUT / "PAPER.md")


if __name__ == "__main__":
    main()

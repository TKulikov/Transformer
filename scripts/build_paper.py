"""Render the scientific article as PAPER.md and a portrait GOST-style PAPER.pdf."""

from __future__ import annotations

import html
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_text as PT
import report_data as RD

COMPETITOR_HEADERS = ["Метод", "Принцип", "Обучение", "Источник и роль"]
COMPETITOR_ROWS = [
    ["Полный контекст", "Вся история и полный KV-кэш", "Нет", "Контрольная точка"],
    ["StreamingLLM", "Attention sinks и недавние токены", "Нет", "[1]; численный контроль"],
    ["H2O", "Heavy hitters по накопленному вниманию", "Нет", "[2]; численный контроль"],
    ["SnapKV", "Отбор префикса по observation-окну", "Нет", "[3]; численный контроль"],
    ["PyramidKV", "Неравномерный бюджет по слоям", "Нет", "[4]; численный контроль"],
    ["KIVI", "Асимметричное 2-битное квантование K и V", "Нет", "[5]; качественный аналог"],
    ["Quest", "Query-aware выбор страниц KV-кэша", "Нет", "[6]; качественный аналог"],
    ["RAG", "Возврат текстовых фрагментов в промпт", "Обычно нет", "[7]; отдельный контроль"],
    ["LongLLMLingua", "Question-aware сжатие промпта", "Нет для базовой модели", "[8]; качественный аналог"],
    ["MemGPT", "Иерархия памяти и функции управления", "Нет обязательно", "[9]; системный аналог"],
    ["MemoryBank", "Поиск, профиль пользователя и забывание", "Нет обязательно", "[10]; системный аналог"],
    ["A-MEM", "Связные заметки и эволюция памяти", "Нет обязательно", "[11]; системный аналог"],
    ["Mem0", "Извлечение, консолидация и графовая память", "Нет обязательно", "[12]; системный аналог"],
    ["Memorizing Transformers", "kNN-поиск по KV прошлых сегментов", "Да", "[13]; архитектурный аналог"],
    ["LongMem", "Поиск и residual side-network", "Да", "[14]; архитектурный аналог"],
    ["RMT", "Memory-токены между сегментами", "Да", "[15]; архитектурный аналог"],
    ["Landmark Attention", "Обучаемые якоря блоков контекста", "Да", "[16]; архитектурный аналог"],
    ["Infini-attention", "Локальное attention и сжатое состояние", "Да", "[17]; архитектурный аналог"],
    ["Memory Layers", "Разреженные обучаемые таблицы памяти", "Да", "[18, 19]; архитектурный аналог"],
    ["Двухканальная память", "Текстовый поиск и дополнительные латентные K/V", "Нет", "Предложенная система"],
]

CONFIG_HEADERS = ["Система", "Контекст", "Память и бюджет"]
CONFIG_ROWS = [
    ["Чистый Qwen", "Вся история", "Классический неограниченный KV-кэш"],
    ["StreamingLLM", "Вся история", "128 слотов: 4 sink + 124 recent"],
    ["H2O", "Вся история", "128: 4 sink + 64 recent + до 60 heavy hitters"],
    ["SnapKV", "Вся история", "128; observation = 64; kernel = 7"],
    ["PyramidKV", "Вся история", r"Среднее 128; observation = 8; $\beta = 20$"],
    ["Текстовый RAG-контроль", "Буфер 10 ходов (5 пар) + top-4", "Только текст; сырые эмбеддинги; классический KV"],
    ["Двухканальная память", "Буфер 10 ходов (5 пар) + top-4", "Текст + центрирование; до 32 K/V в 4 слоях"],
]

VOLUME_HEADERS = ["Система", "Prompt", "Генер.", "Embed", "Attn-слоты", "Экономия", "Max KV"]
LATENCY_HEADERS = ["Система", "Среднее, с", "Медиана, с", "P90, с", "Пик, МиБ"]

REPRO_COMMAND = (
    "HF_HUB_OFFLINE=1 .venv/bin/python -m tmem.cli --model unused --battery100 "
    "--models qwen35-4b --device cuda --dtype bfloat16 --max-new-tokens 32 "
    "--strategies classic_kv_full sparse_sliding_128 sparse_topk_128 snapkv_128 "
    "pyramidkv_128 rag_text_only v4_rag_kv --out results/qwen35_comparison"
)


_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "lambda": "λ", "mu": "μ", "pi": "π", "sigma": "σ", "phi": "φ", "omega": "ω",
}


def _tex_group(tex: str, i: int):
    """Return (argument, next_index) for a LaTeX group or single token at tex[i]."""
    while i < len(tex) and tex[i] == " ":
        i += 1
    if i < len(tex) and tex[i] == "{":
        depth = 0
        for j in range(i, len(tex)):
            if tex[j] == "{":
                depth += 1
            elif tex[j] == "}":
                depth -= 1
                if depth == 0:
                    return tex[i + 1:j], j + 1
        raise ValueError(f"unbalanced braces in formula: {tex}")
    return tex[i], i + 1


def _tex_inner(tex: str, italic: bool = True) -> str:
    """Convert a small inline LaTeX fragment to reportlab paragraph markup.

    Supports the limited grammar used in the article body: variables (italic),
    sub/superscripts, Greek letters, \\mathrm-style commands, and punctuation.
    Display formulas never pass through this function (they are rendered as images).
    """
    out = []
    i = 0
    n = len(tex)
    while i < n:
        ch = tex[i]
        if ch == "\\":
            m = re.match(r"\\([a-zA-Z]+)", tex[i:])
            if m:
                name = m.group(1)
                i += m.end()
                if name in ("mathrm", "text", "operatorname"):
                    arg, i = _tex_group(tex, i)
                    out.append(html.escape(_tex_inner(arg, False)))
                elif name in _GREEK:
                    out.append(_GREEK[name])
                elif name == "cdot":
                    out.append("·")
                elif name == "top":
                    out.append("ᵀ")
                continue
            if i + 1 < n:
                nxt = tex[i + 1]
                if nxt == ",":
                    out.append(" ")
                elif nxt == "!":
                    pass
                elif nxt in (";", " ", ":"):
                    out.append(" ")
                else:
                    out.append(html.escape(nxt))
                i += 2
                continue
            i += 1
            continue
        if ch in "_^":
            tag = "sub" if ch == "_" else "sup"
            arg, i = _tex_group(tex, i + 1)
            out.append(f"<{tag}>{_tex_inner(arg, italic)}</{tag}>")
            continue
        if ch == "{":
            arg, i = _tex_group(tex, i)
            out.append(_tex_inner(arg, italic))
            continue
        if ch == "}":
            i += 1
            continue
        if ch.isalpha():
            out.append(f"<i>{html.escape(ch)}</i>" if italic else html.escape(ch))
            i += 1
            continue
        out.append(html.escape(ch))
        i += 1
    return "".join(out)


def _inline_markup(text: str) -> str:
    """Escape plain text and render $...$ inline math as reportlab markup."""
    parts = re.split(r"\$([^$]+)\$", text)
    out = []
    for k, part in enumerate(parts):
        if k % 2:
            out.append(_tex_inner(part))
        else:
            out.append(html.escape(part).replace("\n", "<br/>"))
    return "".join(out)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _localized_cell(value, column):
    text = str(value)
    if column and re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?×?", text):
        return text.replace(",", " ").replace(".", ",")
    return text


def _localized_rows(rows):
    return [[_localized_cell(cell, column) for column, cell in enumerate(row)] for row in rows]


def _volume_rows(q):
    return [[row[i] for i in (0, 1, 2, 3, 4, 5, 9)] for row in RD.cost_rows(q)]


def _latency_rows(q):
    return [[row[i] for i in (0, 6, 7, 8, 10)] for row in RD.cost_rows(q)]


def build_md(q: dict):
    """Write the Markdown article with numbered LaTeX formulas and GOST-style captions."""
    lines = [
        PT.UDK, "",
        f"# {PT.TITLE}", "", f"_{PT.AUTHORS}_", "", "## АННОТАЦИЯ", "", PT.abstract(q), "",
        f"**Ключевые слова:** {PT.KEYWORDS}", "",
    ]
    formula_number = 0

    def add_text(text):
        nonlocal formula_number
        match = re.fullmatch(r"\$\$\s*(.*?)\s*\$\$", text, re.DOTALL)
        if match:
            formula_number += 1
            lines.extend(["$$", f"{match.group(1)} \\tag{{{formula_number}}}", "$$", ""])
        else:
            lines.extend([text + "\n"])

    for heading, paragraphs in PT.sections(q):
        lines += [f"## {heading}", ""]
        for paragraph in paragraphs:
            add_text(paragraph)
        if heading.startswith("1 "):
            lines += ["**Таблица 1 — Сравнение существующих решений и аналогов**", ""]
            lines += markdown_table(COMPETITOR_HEADERS, COMPETITOR_ROWS) + [""]
        if heading.startswith("2 "):
            lines += ["![Архитектура двухканальной памяти](figs/architecture.png)",
                      "Рисунок 1 — Архитектура предложенной системы", ""]
            lines += ["**Таблица 2 — Параметры сравниваемых систем**", ""]
            lines += markdown_table(CONFIG_HEADERS, CONFIG_ROWS) + [""]
        if heading.startswith("3 "):
            tables = (
                ("Таблица 3 — Качество ответов на контрольные вопросы", RD.QUALITY_HEADERS, RD.quality_rows(q)),
                ("Таблица 4 — Объём входа и decode-side attention-прокси", VOLUME_HEADERS, _volume_rows(q)),
                ("Таблица 5 — Наблюденная латентность и память генерации", LATENCY_HEADERS, _latency_rows(q)),
            )
            for caption, headers, rows in tables:
                lines += [f"**{caption}**", ""] + markdown_table(headers, _localized_rows(rows)) + [""]
            lines += [
                "![Основное сравнение](figs/comparison.png)",
                "Рисунок 2 — Качество, attention-прокси и латентность на общей батарее", "",
                "![Парето](figs/pareto.png)",
                "Рисунок 3 — Компромисс между recall и decode-side attention-прокси", "",
            ]
    lines += [
        "## СПИСОК ЛИТЕРАТУРЫ", "",
        *[f"{i}. {ref}" for i, ref in enumerate(PT.REFERENCES, 1)],
        "", f"## {PT.EN_TITLE}", "", f"_{PT.EN_AUTHORS}_", "",
        "**ABSTRACT**", "", PT.en_abstract(q), "",
        f"**Keywords:** {PT.EN_KEYWORDS}", "",
        "**References**", "",
        *[f"{i}. {ref}" for i, ref in enumerate(PT.REFERENCES, 1)],
        "", "## Команда воспроизведения", "",
        "```bash", REPRO_COMMAND,
        ".venv/bin/python scripts/build_report.py", ".venv/bin/python scripts/build_paper.py", "```",
    ]
    (RD.OUT / "PAPER.md").write_text("\n".join(lines) + "\n")


def build_pdf(q: dict):
    """Write a portrait A4 PDF with Russian typography, numbered formulas, tables and references."""
    import matplotlib
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font_dir = Path("/usr/share/fonts/truetype/liberation")
    pdfmetrics.registerFont(TTFont("LiberationSerif", str(font_dir / "LiberationSerif-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("LiberationSerif-Bold", str(font_dir / "LiberationSerif-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("LiberationSerif-Italic", str(font_dir / "LiberationSerif-Italic.ttf")))
    pdfmetrics.registerFont(TTFont("LiberationSerif-BoldItalic", str(font_dir / "LiberationSerif-BoldItalic.ttf")))
    pdfmetrics.registerFontFamily(
        "LiberationSerif", normal="LiberationSerif", bold="LiberationSerif-Bold",
        italic="LiberationSerif-Italic", boldItalic="LiberationSerif-BoldItalic",
    )
    matplotlib.rcParams["mathtext.fontset"] = "stix"

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "GOSTBody", parent=styles["Normal"], fontName="LiberationSerif", fontSize=14,
        leading=21, alignment=TA_JUSTIFY, firstLineIndent=1.25 * cm, spaceAfter=0,
        splitLongWords=True, hyphenationLang="ru_RU",
    )
    heading = ParagraphStyle(
        "GOSTHeading", parent=body, fontName="LiberationSerif-Bold", leading=18,
        alignment=TA_CENTER, firstLineIndent=0, spaceBefore=12, spaceAfter=12,
        keepWithNext=True,
    )
    table_caption = ParagraphStyle(
        "TableCaption", parent=body, fontSize=12, leading=15, alignment=TA_LEFT,
        firstLineIndent=0, spaceBefore=8, spaceAfter=5, keepWithNext=True,
    )
    figure_caption = ParagraphStyle(
        "FigureCaption", parent=table_caption, alignment=TA_CENTER,
        spaceBefore=5, spaceAfter=10, keepWithNext=False,
    )
    title = ParagraphStyle(
        "TitleGOST", parent=body, fontName="LiberationSerif-Bold", fontSize=16,
        leading=20, alignment=TA_CENTER, firstLineIndent=0, spaceAfter=12,
    )
    authors = ParagraphStyle(
        "Authors", parent=body, fontName="LiberationSerif-Italic", alignment=TA_CENTER,
        firstLineIndent=0, spaceAfter=12,
    )
    table_text = ParagraphStyle(
        "TableText", parent=body, fontSize=10, leading=12, alignment=TA_LEFT,
        firstLineIndent=0, splitLongWords=True,
    )
    table_header = ParagraphStyle(
        "TableHeader", parent=table_text, fontName="LiberationSerif-Bold", alignment=TA_CENTER,
    )
    formula_number_style = ParagraphStyle(
        "FormulaNumber", parent=body, alignment=TA_RIGHT, firstLineIndent=0,
    )
    reference = ParagraphStyle(
        "Reference", parent=body, fontSize=12, leading=16, alignment=TA_LEFT,
        leftIndent=0.9 * cm, firstLineIndent=-0.9 * cm, spaceAfter=5,
    )
    doc = SimpleDocTemplate(
        str(RD.OUT / "PAPER.pdf"), pagesize=A4,
        leftMargin=3.0 * cm, rightMargin=1.5 * cm, topMargin=2.0 * cm, bottomMargin=2.0 * cm,
        title=PT.TITLE, author=PT.AUTHORS, subject="Диалоговая память и сжатие KV-кэша",
    )

    def on_page(canvas, document):
        canvas.saveState()
        canvas.setFont("LiberationSerif", 12)
        canvas.setFillColor(colors.black)
        canvas.drawCentredString(A4[0] / 2, 1.0 * cm, str(document.page))
        canvas.restoreState()

    def paragraph(text, style=body):
        match = re.match(r"^(\d+\.\d+\s+[^.]+\.)\s+(.*)$", text, re.DOTALL)
        if match:
            content = f"<b>{html.escape(match.group(1))}</b> {_inline_markup(match.group(2))}"
            return Paragraph(content, ParagraphStyle("RunIn", parent=style, firstLineIndent=0))
        return Paragraph(_inline_markup(text), style)

    def latex_formula(latex, number):
        buffer = io.BytesIO()
        math_to_image(
            f"${latex}$", buffer, prop=FontProperties(size=12), dpi=300, format="png", color="black",
        )
        buffer.seek(0)
        reader = ImageReader(buffer)
        pixel_width, pixel_height = reader.getSize()
        width = pixel_width * 72 / 300
        height = pixel_height * 72 / 300
        available = doc.width - 1.35 * cm
        scale = min(1.0, available / width)
        buffer.seek(0)
        image = Image(buffer, width=width * scale, height=height * scale)
        image.hAlign = "CENTER"
        table = Table(
            [[image, Paragraph(f"({number})", formula_number_style)]],
            colWidths=[available, 1.35 * cm], hAlign="CENTER",
        )
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        table._formula_buffer = buffer
        return table

    def add_table(story, caption_text, headers, rows, widths, font_size=10):
        story.append(Paragraph(html.escape(caption_text), table_caption))
        cell_style = ParagraphStyle("SizedTableText", parent=table_text, fontSize=font_size, leading=font_size * 1.2)
        head_style = ParagraphStyle("SizedTableHeader", parent=table_header, fontSize=font_size, leading=font_size * 1.2)
        data = [[Paragraph(html.escape(str(cell)), head_style) for cell in headers]]
        data += [
            [Paragraph(_inline_markup(str(cell)), cell_style) for cell in _localized_rows([row])[0]]
            for row in rows
        ]
        table = Table(data, colWidths=[width * cm for width in widths], repeatRows=1, hAlign="LEFT", splitByRow=1)
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story += [table, Spacer(1, 0.2 * cm)]

    def figure(path, caption_text, width=16.0 * cm):
        reader = ImageReader(str(path))
        pixel_width, pixel_height = reader.getSize()
        image = Image(str(path), width=width, height=width * pixel_height / pixel_width)
        image.hAlign = "CENTER"
        return KeepTogether([image, Paragraph(html.escape(caption_text), figure_caption)])

    udk_style = ParagraphStyle("Udk", parent=body, alignment=TA_LEFT, firstLineIndent=0, spaceAfter=10)
    story = [
        Paragraph(html.escape(PT.UDK), udk_style),
        Paragraph(html.escape(PT.TITLE), title), Paragraph(html.escape(PT.AUTHORS), authors),
        Paragraph("АННОТАЦИЯ", heading), Paragraph(html.escape(PT.abstract(q)), body),
        Paragraph(f"<b>Ключевые слова:</b> {html.escape(PT.KEYWORDS)}", body),
    ]
    formula_number = 0
    for section, paragraphs in PT.sections(q):
        story.append(Paragraph(html.escape(section), heading))
        for text in paragraphs:
            match = re.fullmatch(r"\$\$\s*(.*?)\s*\$\$", text, re.DOTALL)
            if match:
                formula_number += 1
                story.append(latex_formula(match.group(1), formula_number))
            else:
                story.append(paragraph(text))
        if section.startswith("1 "):
            add_table(story, "Таблица 1 — Сравнение существующих решений и аналогов",
                      COMPETITOR_HEADERS, COMPETITOR_ROWS, [3.2, 6.0, 2.3, 5.0])
        if section.startswith("2 "):
            story.append(figure(RD.FIGS / "architecture.png", "Рисунок 1 — Архитектура предложенной системы"))
            add_table(story, "Таблица 2 — Параметры сравниваемых систем",
                      CONFIG_HEADERS, CONFIG_ROWS, [4.0, 3.7, 8.8])
        if section.startswith("3 "):
            add_table(story, "Таблица 3 — Качество ответов на контрольные вопросы",
                      RD.QUALITY_HEADERS, RD.quality_rows(q), [5.1, 1.8, 1.8, 2.2, 1.8, 1.8, 2.0], 9)
            add_table(story, "Таблица 4 — Объём входа и decode-side attention-прокси",
                      VOLUME_HEADERS, _volume_rows(q), [4.2, 2.0, 1.7, 1.7, 2.8, 1.8, 2.3], 9)
            add_table(story, "Таблица 5 — Наблюденная латентность и память генерации",
                      LATENCY_HEADERS, _latency_rows(q), [6.0, 2.5, 2.5, 2.5, 3.0], 10)
            story.append(figure(RD.FIGS / "comparison.png", "Рисунок 2 — Качество, attention-прокси и латентность на общей батарее"))
            story.append(figure(RD.FIGS / "pareto.png", "Рисунок 3 — Компромисс между recall и decode-side attention-прокси", 15.0 * cm))
    story.append(Paragraph("СПИСОК ЛИТЕРАТУРЫ", heading))
    story += [
        Paragraph(html.escape(f"{i}. {ref}"), reference)
        for i, ref in enumerate(PT.REFERENCES, 1)
    ]
    story.append(Paragraph(html.escape(PT.EN_TITLE), heading))
    story.append(Paragraph(html.escape(PT.EN_AUTHORS), authors))
    story.append(Paragraph("ABSTRACT", heading))
    story.append(Paragraph(html.escape(PT.en_abstract(q)), body))
    story.append(Paragraph(f"<b>Keywords:</b> {html.escape(PT.EN_KEYWORDS)}", body))
    story.append(Paragraph("REFERENCES", heading))
    story += [
        Paragraph(html.escape(f"{i}. {ref}"), reference)
        for i, ref in enumerate(PT.REFERENCES, 1)
    ]
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


SKILL_TYPO = Path("/home/morgot/Documents/Skills/gost-typography")


def _number_formulas_in_docx(path):
    """Wrap every display-math paragraph into a borderless 2-column table (formula | number).

    pandoc drops \\tag{n} when converting LaTeX math to OMML, so ГОСТ-style formula
    numbers (right margin, level of the formula) are restored by post-processing.
    """
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    doc = Document(str(path))
    omath_para = qn("m:oMathPara")
    math_paras = [p for p in doc.paragraphs if p._p.find(".//" + omath_para) is not None]
    for idx, para in enumerate(math_paras, 1):
        pe = para._p
        tbl = OxmlElement("w:tbl")
        tbl_pr = OxmlElement("w:tblPr")
        tbl_w = OxmlElement("w:tblW")
        tbl_w.set(qn("w:type"), "pct")
        tbl_w.set(qn("w:w"), "5000")
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "none")
            el.set(qn("w:sz"), "0")
            borders.append(el)
        tbl_pr.append(tbl_w)
        tbl_pr.append(borders)
        tbl.append(tbl_pr)
        grid = OxmlElement("w:tblGrid")
        for w in ("8300", "1030"):
            gc = OxmlElement("w:gridCol")
            gc.set(qn("w:w"), w)
            grid.append(gc)
        tbl.append(grid)
        tr = OxmlElement("w:tr")
        cells = []
        for w in ("8300", "1030"):
            tc = OxmlElement("w:tc")
            tc_pr = OxmlElement("w:tcPr")
            tc_w = OxmlElement("w:tcW")
            tc_w.set(qn("w:type"), "dxa")
            tc_w.set(qn("w:w"), w)
            valign = OxmlElement("w:vAlign")
            valign.set(qn("w:val"), "center")
            tc_pr.append(tc_w)
            tc_pr.append(valign)
            tc.append(tc_pr)
            cells.append(tc)
        pe.addprevious(tbl)
        cells[0].append(pe)
        np = OxmlElement("w:p")
        p_pr = OxmlElement("w:pPr")
        jc = OxmlElement("w:jc")
        jc.set(qn("w:val"), "right")
        p_pr.append(jc)
        np.append(p_pr)
        run = OxmlElement("w:r")
        text = OxmlElement("w:t")
        text.text = f"({idx})"
        run.append(text)
        np.append(run)
        cells[1].append(np)
        tr.append(cells[0])
        tr.append(cells[1])
        tbl.append(tr)
    doc.save(str(path))
    return len(math_paras)


def build_docx():
    """Build PAPER.docx via pandoc with the GOST reference doc from the gost-typography skill."""
    import shutil
    import subprocess

    pandoc = shutil.which("pandoc") or str(Path.home() / ".local/bin" / "pandoc")
    if not Path(pandoc).exists():
        print("pandoc не найден — DOCX пропущен")
        return None
    tmp = Path("/tmp/opencode")
    tmp.mkdir(parents=True, exist_ok=True)
    base, ref = tmp / "ref-base.docx", tmp / "gost-ref.docx"
    subprocess.run([pandoc, "-o", str(base), "--print-default-data-file", "reference.docx"], check=True)
    subprocess.run([sys.executable, str(SKILL_TYPO / "scripts" / "gost_reference_doc.py"),
                    "--base", str(base), "-o", str(ref)], check=True)
    md = (RD.OUT / "PAPER.md").read_text()
    md = md.split("## Команда воспроизведения")[0].rstrip() + "\n"
    docx_md = tmp / "PAPER-docx.md"
    docx_md.write_text(md)
    out = RD.OUT / "PAPER.docx"
    subprocess.run([pandoc, str(docx_md), "-o", str(out), f"--reference-doc={ref}",
                    f"--resource-path={RD.OUT}"], check=True)
    print("формул пронумеровано:", _number_formulas_in_docx(out))
    subprocess.run([sys.executable, str(SKILL_TYPO / "scripts" / "check_docx.py"), str(out)], check=True)
    return out


def main():
    q = RD.q35_all()
    missing = set(RD.ORDER) - set(q)
    if missing:
        raise SystemExit(f"не хватает результатов: {sorted(missing)}")
    import paper_figures
    paper_figures.make_all()
    build_md(q)
    build_pdf(q)
    build_docx()
    print("paper:", RD.OUT / "PAPER.pdf")


if __name__ == "__main__":
    main()

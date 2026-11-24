"""Render the unified scientific article as PAPER.md and standard-layout PAPER.pdf."""

from __future__ import annotations

import html
import io
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paper_text as PT
import report_data as RD

COMPETITOR_HEADERS = ["Метод", "Принцип", "Обучение", "Роль в работе"]
COMPETITOR_ROWS = [
    ["Полный контекст", "Вся история и полный KV-кэш", "Нет", "Верхний контроль"],
    ["StreamingLLM", "Attention sinks и недавние токены", "Нет", "[1]; численный контроль"],
    ["H2O / TopK", "Heavy hitters по накопленному вниманию", "Нет", "[2]; численный контроль"],
    ["SnapKV", "Отбор префикса по observation-окну", "Нет", "[3]; численный контроль"],
    ["PyramidKV", "Неравномерный бюджет по слоям", "Нет", "[4]; численный контроль"],
    ["KIVI", "Асимметричное 2-битное квантование K/V", "Нет", "[5]; качественный аналог"],
    ["Quest", "Query-aware выбор страниц KV-кэша", "Нет", "[6]; качественный аналог"],
    ["RAG", "Возврат текстовых фрагментов в промпт", "Обычно нет", "[7]; отдельный канал"],
    ["LongLLMLingua", "Question-aware сжатие промпта", "Нет для backbone", "[8]; качественный аналог"],
    ["MemGPT", "Иерархия памяти и функции управления", "Не обязательно", "[9]; системный аналог"],
    ["MemoryBank", "Поиск, профиль и забывание", "Не обязательно", "[10]; системный аналог"],
    ["A-MEM", "Связные заметки и эволюция памяти", "Не обязательно", "[11]; системный аналог"],
    ["Mem0", "Извлечение, консолидация и граф", "Не обязательно", "[12]; системный аналог"],
    ["Memorizing Transformers", "kNN-поиск по K/V прошлых сегментов", "Да", "[13]; близкий архитектурный аналог"],
    ["Memory³", "Явная внешняя память из sparse K/V", "Да", "[30]; близкий архитектурный аналог"],
    ["LongMem", "Поиск и residual side-network", "Да", "[14]; архитектурный аналог"],
    ["RMT", "Memory-токены между сегментами", "Да", "[15]; архитектурный аналог"],
    ["Landmark Attention", "Обучаемые якоря блоков", "Да", "[16]; архитектурный аналог"],
    ["Infini-attention", "Локальное attention и сжатое состояние", "Да", "[17]; архитектурный аналог"],
    ["Memory Layers", "Разреженные обучаемые таблицы памяти", "Да", "[18, 19]; архитектурный аналог"],
    ["M+ / MemoryLLM", "Самообновляемый латентный пул", "Да", "[33]; аналог скрытых состояний"],
    ["MemDefrag", "Консолидация и забывание latent-памяти", "Зависит от backbone", "[34]; аналог динамики памяти"],
    ["Живая память", "TTL/EMA-банк + динамическая K/V-инъекция", "Нет", "Предложенный вариант"],
]

RUN_HEADERS = ["Серия", "Диалоги", "Назначение", "Валидность"]
RUN_ROWS = [
    ["battery100", "1 × 107 ходов", "Предварительное сравнение 7 систем", "pre-fix; только исторический контекст"],
    ["latent-v1", "4/12 × 100 ходов", "Mid-RAG, RAG±K/V, inject-only", "post-fix; пилот"],
    ["full-core-v1", "5/48 × 100 ходов", "Конкуренты и k32 vs dynamic", "post-fix; основной срез"],
    ["full-rag-v1", "4/48 × 100 ходов", "Last/mid/mean × raw/centered × ±K/V", "post-fix; основной срез"],
    ["Специальные тесты", "Малые серии", "TTL/gist, цель, нити, слепая оценка", "Диагностические; разные протоколы"],
]

CONFIG_HEADERS = ["Система", "Контекст", "Память и параметры"]
CONFIG_ROWS = [
    ["Full KV", "Вся история", "Неограниченный классический KV-кэш"],
    ["Sliding / TopK", "Вся история", "128 слотов; 4 sink + recent/heavy hitters"],
    ["SnapKV", "Вся история", "128; observation=64; kernel=7"],
    ["PyramidKV", "Вся история", "Среднее 128; observation=8; β=20"],
    ["RAG-only", "10 последних пар + top-4", "Last/mid/mean; raw/centered; банк 256"],
    ["Inject-only k32", "10 последних пар", "Банк 512; до 32 K/V в 4 слоях; без текста"],
    ["Inject-only dynamic", "10 последних пар", "Косинусный гейт 0,5; 0–32 K/V; без текста"],
    ["RAG + K/V", "10 последних пар + top-4", "Текстовый канал + фиксированная k32"],
]

MECHANISM_HEADERS = ["Тест", "Условие", "Наблюдение", "Ограничение"]
MECHANISM_ROWS = [
    ["goal_dialogue", "Merge-цель vs чистый centered RAG", "Confusion 0,600 vs 0,400", "Специальный малый тест"],
    ["interlocutor", "Gist vs plain RAG", "0,756 vs 0,760; метрика насыщена", "Низкая чувствительность"],
    ["thread_coherence", "K/V после фикса маски", "Confusion 0,286 vs 0,000 у v3", "7 проб; другой протокол"],
    ["blind_eval", "Слепая оценка 5 условий", "Нет консенсуса о преимуществе", "Субъективная оценка"],
    ["living_memory", "Явный TTL-gist поверх центрирования", "Дополнительного выигрыша нет", "Отличать от running centroid"],
]

CORE_HEADERS = ["Система", "Recall [95% ДИ]", "Answerable", "E2E, с/ход", "Peak MiB", "Prompt"]
HYBRID_HEADERS = ["Канал", "RAG-only", "+K/V", "Δ recall", "p Холма", "Δ E2E"]
REPRO_COMMANDS = [
    "scripts/run_dialogue_fullscale.sh core night",
    "scripts/run_dialogue_fullscale.sh rag night",
    ".venv/bin/python scripts/build_core_report.py",
    ".venv/bin/python scripts/build_rag_only_report.py",
    ".venv/bin/python scripts/build_paper.py",
    ".venv/bin/python scripts/verify_paper.py",
]

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "lambda": "λ", "mu": "μ", "pi": "π", "sigma": "σ", "phi": "φ", "omega": "ω",
}


def _tex_group(tex: str, i: int):
    """Return a LaTeX group or token and the following input index."""
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
    """Convert the article's inline LaTeX subset to ReportLab paragraph markup."""
    out = []
    i = 0
    while i < len(tex):
        ch = tex[i]
        if ch == "\\":
            match = re.match(r"\\([a-zA-Z]+)", tex[i:])
            if match:
                name = match.group(1)
                i += match.end()
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
            if i + 1 < len(tex):
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
    """Escape plain text and convert inline $...$ mathematics for ReportLab."""
    parts = re.split(r"\$([^$]+)\$", text)
    out = []
    for index, part in enumerate(parts):
        out.append(_tex_inner(part) if index % 2 else html.escape(part).replace("\n", "<br/>"))
    return "".join(out)


def markdown_table(headers, rows):
    """Return a Markdown table without altering scientific identifiers."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def _article_tables():
    """Load all current tables once for consistent Markdown and PDF output."""
    core = RD.core_data()
    rag = RD.rag_data()
    return core, rag, RD.core_table(core), RD.rag_pure_table(rag), RD.rag_hybrid_table(rag)


def build_md(q: dict):
    """Write the unified Markdown article with numbered formulas and current figures."""
    core, rag, core_rows, rag_rows, hybrid_rows = _article_tables()
    lines = [
        f"# {PT.TITLE}", "",
        f"*Научная статья · экспериментальный срез: core {len(core['records'])}/48, "
        f"RAG {len(rag['records'])}/48 · {date.today().isoformat()}*", "",
        "## АННОТАЦИЯ", "", PT.abstract(q), "",
        f"**Ключевые слова:** {PT.KEYWORDS}", "",
        f"## {PT.EN_TITLE}", "", "**ABSTRACT**", "", PT.en_abstract(q), "",
        f"**Keywords:** {PT.EN_KEYWORDS}", "",
    ]
    formula_number = 0
    for heading, paragraphs in PT.sections(q):
        lines += [f"## {heading}", ""]
        for paragraph in paragraphs:
            match = re.fullmatch(r"\$\$\s*(.*?)\s*\$\$", paragraph, re.DOTALL)
            if match:
                formula_number += 1
                lines += ["$$", f"{match.group(1)} \\tag{{{formula_number}}}", "$$", ""]
            else:
                lines += [paragraph + "\n"]
        if heading.startswith("1 "):
            lines += ["**Таблица 1 — Существующие решения и аналоги**", ""]
            lines += markdown_table(COMPETITOR_HEADERS, COMPETITOR_ROWS) + [""]
        elif heading.startswith("2 "):
            lines += ["![Архитектура](figs/architecture.png)",
                      "Рисунок 1 — Независимые текстовый и латентный каналы", ""]
        elif heading.startswith("3 "):
            for caption, headers, rows in [
                ("Таблица 2 — Экспериментальные серии", RUN_HEADERS, RUN_ROWS),
                ("Таблица 3 — Параметры сравниваемых систем", CONFIG_HEADERS, CONFIG_ROWS),
            ]:
                lines += [f"**{caption}**", ""] + markdown_table(headers, rows) + [""]
        elif heading.startswith("4 "):
            lines += ["**Таблица 4 — Живая память и конкуренты на core**", ""]
            lines += markdown_table(CORE_HEADERS, core_rows) + [""]
            for filename, caption in [
                ("core_quality_cost.png", "Рисунок 2 — Core: качество, E2E и peak allocated VRAM"),
                ("core_per_dialogue.png", "Рисунок 3 — Core: recall по отдельным диалогам"),
                ("core_lag.png", "Рисунок 4 — Core: удержание по lag-зонам"),
            ]:
                lines += [f"![{caption}](figs/{filename})", caption, ""]
            lines += ["**Таблица 5 — Текстовый RAG без K/V-инъекций**", ""]
            lines += markdown_table(CORE_HEADERS, rag_rows) + [""]
            for filename, caption in [
                ("rag_only_quality_cost.png", "Рисунок 5 — RAG-only: качество и стоимость"),
                ("rag_only_layer_centering.png", "Рисунок 6 — Парный эффект центрирования"),
                ("rag_only_per_dialogue.png", "Рисунок 7 — RAG-only по диалогам"),
                ("rag_only_lag.png", "Рисунок 8 — RAG-only по lag-зонам"),
            ]:
                lines += [f"![{caption}](figs/{filename})", caption, ""]
            lines += ["**Таблица 6 — Вклад фиксированной K/V-инъекции в RAG**", ""]
            lines += markdown_table(HYBRID_HEADERS, hybrid_rows) + [""]
            lines += ["![RAG и K/V](figs/rag_hybrid_pairs.png)",
                      "Рисунок 9 — Сравнение шести пар RAG-only и RAG+K/V", ""]
            lines += ["**Таблица 7 — Дополнительные механистические тесты**", ""]
            lines += markdown_table(MECHANISM_HEADERS, MECHANISM_ROWS) + [""]
    lines += ["## СПИСОК ЛИТЕРАТУРЫ", ""]
    lines += [f"{index}. {ref}" for index, ref in enumerate(PT.REFERENCES, 1)]
    lines += ["", "## ВОСПРОИЗВЕДЕНИЕ", "", f"Репозиторий проекта: <{PT.REPOSITORY_URL}>", "", "```bash", *REPRO_COMMANDS, "```"]
    (RD.OUT / "PAPER.md").write_text("\n".join(lines) + "\n")


def build_pdf(q: dict):
    """Write a standard portrait A4 scientific PDF without GOST-specific layout."""
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
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font_dir = Path("/usr/share/fonts/truetype/liberation")
    pdfmetrics.registerFont(TTFont("Serif", str(font_dir / "LiberationSerif-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("Serif-Bold", str(font_dir / "LiberationSerif-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Serif-Italic", str(font_dir / "LiberationSerif-Italic.ttf")))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic")
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    styles = getSampleStyleSheet()
    body = ParagraphStyle("ArticleBody", parent=styles["Normal"], fontName="Serif", fontSize=10.8,
                          leading=15, alignment=TA_JUSTIFY, firstLineIndent=0.65 * cm, spaceAfter=5,
                          splitLongWords=True, hyphenationLang="ru_RU")
    heading = ParagraphStyle("ArticleHeading", parent=body, fontName="Serif-Bold", fontSize=13,
                             leading=16, alignment=TA_CENTER, firstLineIndent=0, spaceBefore=12,
                             spaceAfter=8, keepWithNext=True)
    title = ParagraphStyle("ArticleTitle", parent=body, fontName="Serif-Bold", fontSize=17,
                           leading=21, alignment=TA_CENTER, firstLineIndent=0, spaceAfter=10)
    subtitle = ParagraphStyle("ArticleSubtitle", parent=body, fontName="Serif-Italic", fontSize=10,
                              alignment=TA_CENTER, firstLineIndent=0, textColor=colors.HexColor("#475569"))
    table_caption = ParagraphStyle("ArticleTableCaption", parent=body, fontSize=9.5, leading=12,
                                   alignment=TA_LEFT, firstLineIndent=0, spaceBefore=8, spaceAfter=5,
                                   keepWithNext=True)
    figure_caption = ParagraphStyle("ArticleFigureCaption", parent=table_caption, alignment=TA_CENTER,
                                    spaceBefore=4, spaceAfter=9, keepWithNext=False)
    table_text = ParagraphStyle("ArticleTableText", parent=body, fontSize=8.2, leading=10,
                                alignment=TA_LEFT, firstLineIndent=0, splitLongWords=True)
    table_header = ParagraphStyle("ArticleTableHeader", parent=table_text, fontName="Serif-Bold",
                                  alignment=TA_CENTER)
    formula_number_style = ParagraphStyle("FormulaNumber", parent=body, alignment=TA_RIGHT, firstLineIndent=0)
    reference = ParagraphStyle("ArticleReference", parent=body, fontSize=9.5, leading=12.5,
                               alignment=TA_LEFT, leftIndent=0.7 * cm, firstLineIndent=-0.7 * cm, spaceAfter=4)
    code = ParagraphStyle("ArticleCode", parent=body, fontName="Courier", fontSize=8.2, leading=10,
                          firstLineIndent=0, leftIndent=0.4 * cm)

    core, rag, core_rows, rag_rows, hybrid_rows = _article_tables()
    doc = SimpleDocTemplate(str(RD.OUT / "PAPER.pdf"), pagesize=A4, leftMargin=2.0 * cm,
                            rightMargin=2.0 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title=PT.TITLE, author="Transformer experiments",
                            subject="Живая латентная память и текстовый retrieval")

    def on_page(canvas, document):
        canvas.saveState()
        canvas.setFont("Serif", 9)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawCentredString(A4[0] / 2, 0.8 * cm, str(document.page))
        canvas.restoreState()

    def paragraph(text, style=body):
        match = re.match(r"^(\d+\.\d+\s+[^.]+\.)\s+(.*)$", text, re.DOTALL)
        if match:
            content = f"<b>{html.escape(match.group(1))}</b> {_inline_markup(match.group(2))}"
            return Paragraph(content, ParagraphStyle("RunIn", parent=style, firstLineIndent=0))
        return Paragraph(_inline_markup(text), style)

    def latex_formula(latex, number):
        buffer = io.BytesIO()
        math_to_image(f"${latex}$", buffer, prop=FontProperties(size=12), dpi=300,
                      format="png", color="black")
        buffer.seek(0)
        reader = ImageReader(buffer)
        pixel_width, pixel_height = reader.getSize()
        width = pixel_width * 72 / 300
        height = pixel_height * 72 / 300
        available = doc.width - 1.1 * cm
        scale = min(1.0, available / width)
        buffer.seek(0)
        image = Image(buffer, width=width * scale, height=height * scale)
        image.hAlign = "CENTER"
        table = Table([[image, Paragraph(f"({number})", formula_number_style)]],
                      colWidths=[available, 1.1 * cm], hAlign="CENTER")
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        table._formula_buffer = buffer
        return table

    def add_table(story, caption_text, headers, rows, widths=None, font_size=8.2):
        story.append(Paragraph(html.escape(caption_text), table_caption))
        cell_style = ParagraphStyle("SizedArticleCell", parent=table_text, fontSize=font_size,
                                    leading=font_size * 1.2)
        head_style = ParagraphStyle("SizedArticleHead", parent=table_header, fontSize=font_size,
                                    leading=font_size * 1.2)
        data = [[Paragraph(html.escape(str(cell)), head_style) for cell in headers]]
        data += [[Paragraph(_inline_markup(str(cell)), cell_style) for cell in row] for row in rows]
        if widths is None:
            first = min(4.0, 17.0 / len(headers) * 1.5)
            widths = [first] + [(17.0 - first) / (len(headers) - 1)] * (len(headers) - 1)
        table = Table(data, colWidths=[width * cm for width in widths], repeatRows=1,
                      hAlign="CENTER", splitByRow=1)
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#64748b")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story += [table, Spacer(1, 0.18 * cm)]

    def figure(path, caption_text, width=16.8 * cm):
        reader = ImageReader(str(path))
        pixel_width, pixel_height = reader.getSize()
        image = Image(str(path), width=width, height=width * pixel_height / pixel_width)
        image.hAlign = "CENTER"
        return KeepTogether([image, Paragraph(html.escape(caption_text), figure_caption)])

    story = [
        Paragraph(html.escape(PT.TITLE), title),
        Paragraph(f"Научная статья · core {len(core['records'])}/48 · RAG {len(rag['records'])}/48 · "
                  f"{date.today().isoformat()}", subtitle),
        Paragraph("АННОТАЦИЯ", heading), Paragraph(html.escape(PT.abstract(q)), body),
        Paragraph(f"<b>Ключевые слова:</b> {html.escape(PT.KEYWORDS)}", body),
        Paragraph(html.escape(PT.EN_TITLE), heading), Paragraph("ABSTRACT", heading),
        Paragraph(html.escape(PT.en_abstract(q)), body),
        Paragraph(f"<b>Keywords:</b> {html.escape(PT.EN_KEYWORDS)}", body), PageBreak(),
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
            add_table(story, "Таблица 1 — Существующие решения и аналоги",
                      COMPETITOR_HEADERS, COMPETITOR_ROWS, [3.4, 6.0, 2.5, 5.1], 7.5)
        elif section.startswith("2 "):
            story.append(figure(RD.FIGS / "architecture.png",
                                "Рисунок 1 — Независимые текстовый и латентный каналы"))
        elif section.startswith("3 "):
            add_table(story, "Таблица 2 — Экспериментальные серии", RUN_HEADERS, RUN_ROWS,
                      [3.1, 3.0, 6.0, 4.9], 7.7)
            add_table(story, "Таблица 3 — Параметры сравниваемых систем", CONFIG_HEADERS,
                      CONFIG_ROWS, [4.0, 4.0, 9.0], 8.0)
        elif section.startswith("4 "):
            add_table(story, "Таблица 4 — Живая память и конкуренты на core", CORE_HEADERS,
                      core_rows, [4.2, 3.6, 2.1, 2.2, 2.0, 2.9], 7.7)
            for filename, caption in [
                ("core_quality_cost.png", "Рисунок 2 — Core: качество, E2E и peak allocated VRAM"),
                ("core_per_dialogue.png", "Рисунок 3 — Core: recall по отдельным диалогам"),
                ("core_lag.png", "Рисунок 4 — Core: удержание по lag-зонам"),
            ]:
                story.append(figure(RD.FIGS / filename, caption))
            add_table(story, "Таблица 5 — Текстовый RAG без K/V-инъекций", CORE_HEADERS,
                      rag_rows, [4.2, 3.6, 2.1, 2.2, 2.0, 2.9], 7.7)
            for filename, caption in [
                ("rag_only_quality_cost.png", "Рисунок 5 — RAG-only: качество и стоимость"),
                ("rag_only_layer_centering.png", "Рисунок 6 — Парный эффект центрирования"),
                ("rag_only_per_dialogue.png", "Рисунок 7 — RAG-only по диалогам"),
                ("rag_only_lag.png", "Рисунок 8 — RAG-only по lag-зонам"),
            ]:
                story.append(figure(RD.FIGS / filename, caption))
            add_table(story, "Таблица 6 — Вклад фиксированной K/V-инъекции в RAG",
                      HYBRID_HEADERS, hybrid_rows, [3.7, 2.2, 2.0, 2.4, 2.3, 2.2], 8.0)
            story.append(figure(RD.FIGS / "rag_hybrid_pairs.png",
                                "Рисунок 9 — Сравнение шести пар RAG-only и RAG+K/V"))
            add_table(story, "Таблица 7 — Дополнительные механистические тесты",
                      MECHANISM_HEADERS, MECHANISM_ROWS, [3.2, 5.0, 5.0, 3.8], 7.5)
    story.append(Paragraph("СПИСОК ЛИТЕРАТУРЫ", heading))
    story += [Paragraph(html.escape(f"{index}. {ref}"), reference)
              for index, ref in enumerate(PT.REFERENCES, 1)]
    story.append(Paragraph("ВОСПРОИЗВЕДЕНИЕ", heading))
    story.append(Paragraph(
        f'Репозиторий проекта: <link href="{PT.REPOSITORY_URL}">{PT.REPOSITORY_URL}</link>', reference
    ))
    story += [Paragraph(html.escape(command), code) for command in REPRO_COMMANDS]
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main():
    """Regenerate figures, Markdown, and PDF from all current experimental archives."""
    q = RD.q35_all()
    missing = set(RD.ORDER) - set(q)
    if missing:
        raise SystemExit(f"не хватает результатов: {sorted(missing)}")
    import paper_figures
    paper_figures.make_all()
    build_md(q)
    build_pdf(q)
    print("paper:", RD.OUT / "PAPER.md")
    print("paper:", RD.OUT / "PAPER.pdf")


if __name__ == "__main__":
    main()

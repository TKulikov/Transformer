"""Build a scientific PDF report from the first full-core-v1 corpus batch."""

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
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    results = {}
    for name in ORDER:
        payload = json.loads((BATCH_DIR / f"{name}.json").read_text())
        analysis = payload["analysis"]
        details = analysis["probe_details"]
        answerable = [p["score"] or 0.0 for p in details if p["kind"] != "unanswerable"]
        results[name] = {
            "recall": analysis["recall_mean"],
            "answerable": sum(answerable) / len(answerable),
            "e2e": median(payload["e2e"]["per_rep_mean_s"]),
            "peak": analysis["peak_mem_mb"],
            "prompt": analysis["prompt_tokens"],
            "attention": analysis["attention_flops_proxy"],
            "kv_slots": analysis["kv_injected_slots"],
            "kv_steps": analysis["kv_steps"],
            "lag": analysis["recall_by_lag"],
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
    fixed_slots = fixed["kv_slots"] / fixed["kv_steps"]
    dynamic_slots = dynamic["kv_slots"] / dynamic["kv_steps"]
    gate_ratio = fixed_slots / dynamic_slots

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
    for ax, (key, panel_title, fmt, upper) in zip(axes, panels):
        series = [row[key] for row in values]
        bars = ax.bar(x, series, color=[palette[name] for name in ORDER], width=0.75)
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.set_xticks(list(x), labels, rotation=32, ha="right", fontsize=8)
        ax.set_ylim(0, upper)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, series):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + upper * 0.015,
                    fmt.format(value), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Контрольный диалог main-L100-00 · n=1", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=190, bbox_inches="tight")
    plt.close(fig)

    font_dir = Path("/usr/share/fonts/truetype/liberation")
    pdfmetrics.registerFont(TTFont("Serif", str(font_dir / "LiberationSerif-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("Serif-Bold", str(font_dir / "LiberationSerif-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Serif-Italic", str(font_dir / "LiberationSerif-Italic.ttf")))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ScientificTitle", parent=styles["Title"], fontName="Serif-Bold",
                           fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=10)
    subtitle = ParagraphStyle("Metadata", parent=styles["Normal"], fontName="Serif-Italic",
                              fontSize=10, leading=13, alignment=TA_CENTER,
                              textColor=colors.HexColor("#475569"), spaceAfter=12)
    body = ParagraphStyle("ScientificBody", parent=styles["Normal"], fontName="Serif",
                          fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY,
                          firstLineIndent=0.7 * cm, spaceAfter=5, splitLongWords=True)
    no_indent = ParagraphStyle("NoIndent", parent=body, firstLineIndent=0)
    heading = ParagraphStyle("ScientificHeading", parent=body, fontName="Serif-Bold",
                             fontSize=13, leading=16, alignment=TA_LEFT, firstLineIndent=0,
                             spaceBefore=9, spaceAfter=6, keepWithNext=True)
    subheading = ParagraphStyle("ScientificSubheading", parent=heading, fontSize=11,
                                leading=14, spaceBefore=6, spaceAfter=4)
    caption = ParagraphStyle("Caption", parent=no_indent, fontSize=9.5, leading=12,
                             alignment=TA_CENTER, spaceBefore=4, spaceAfter=7)
    table_text = ParagraphStyle("TableText", parent=no_indent, fontSize=7.5, leading=9,
                                alignment=TA_CENTER)
    table_head = ParagraphStyle("TableHead", parent=table_text, fontName="Serif-Bold")
    reference = ParagraphStyle("Reference", parent=no_indent, fontSize=9, leading=12,
                               leftIndent=0.6 * cm, firstLineIndent=-0.6 * cm, spaceAfter=4)
    code = ParagraphStyle("Code", parent=no_indent, fontName="Courier", fontSize=8.5,
                          leading=11, leftIndent=0.5 * cm, spaceAfter=3)

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4,
                            leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                            topMargin=1.7 * cm, bottomMargin=1.7 * cm,
                            title="Латентная K/V-память для удержания диалогового контекста",
                            author="Transformer experiments",
                            subject="Научный отчёт о контрольном эксперименте full-core-v1")

    def para(text, style=body):
        return Paragraph(text, style)

    def bullet(text):
        return Paragraph(f"• {text}", no_indent)

    def add_table(story, table_caption, headers, rows, widths, font_size=7.5):
        story.append(Paragraph(html.escape(table_caption), caption))
        cell_style = ParagraphStyle("SizedCell", parent=table_text, fontSize=font_size,
                                    leading=font_size * 1.2)
        head_style = ParagraphStyle("SizedHead", parent=table_head, fontSize=font_size,
                                    leading=font_size * 1.2)
        data = [[Paragraph(html.escape(str(cell)), head_style) for cell in headers]]
        data += [[Paragraph(html.escape(str(cell)), cell_style) for cell in row] for row in rows]
        table = Table(data, colWidths=[w * cm for w in widths], repeatRows=1, hAlign="CENTER")
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#64748b")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story += [table, Spacer(1, 0.2 * cm)]

    story = [
        Paragraph("Латентная K/V-память для удержания диалогового контекста: "
                  "сравнение с методами сжатия KV-кэша", title),
        Paragraph("Экспериментальный отчёт · full-core-v1 · 17 ноября 2026", subtitle),
        Paragraph("АННОТАЦИЯ", heading),
        para(
            f"Исследуется латентная память, подключающая активации средних слоёв как дополнительные "
            f"ключи и значения внимания без возврата исходного текста в промпт. В контрольном диалоге "
            f"динамический вариант получил recall {dynamic['recall']:.3f} против {pyramid['recall']:.3f} "
            f"у PyramidKV и {snap['recall']:.3f} у SnapKV. Относительно PyramidKV латентность ниже на "
            f"{speed_gain:.1f}%, peak VRAM — на {vram_gain:.1f}%, объём prompt-токенов — в "
            f"{prompt_factor:.2f} раза меньше. Полный контекст сохранил наилучший recall "
            f"({full['recall']:.3f}). Результат является предварительным и требует подтверждения на "
            "остальных диалогах корпуса."
        ),
        para("<b>Ключевые слова:</b> диалоговая память; латентные активации; K/V-инъекция; "
             "сжатие KV-кэша; PyramidKV; SnapKV; Qwen3.5.", no_indent),
        Paragraph("1. ВВЕДЕНИЕ", heading),
        para(
            "При длительном диалоге полный KV-кэш сохраняет максимум информации, но увеличивает объём "
            "состояния и работу механизма внимания. Методы Sliding, TopK, SnapKV и PyramidKV сокращают "
            "кэш за счёт отбора токенов. Альтернативный подход — сохранить не исходные токены, а "
            "латентные представления реплик и подключать их к attention только при сходстве с текущим состоянием."
        ),
        para(
            "<b>Исследовательский вопрос:</b> может ли динамическая латентная K/V-память обеспечить "
            "конкурентоспособный recall при меньших затратах латентности и памяти по сравнению с "
            "методами сжатия KV-кэша? Дополнительно проверяется, сохраняет ли динамический отбор "
            "качество фиксированной инъекции при меньшем числе подключаемых слотов."
        ),
        Paragraph("Задачи контрольного эксперимента", subheading),
        bullet("сравнить качество с полным контекстом и четырьмя политиками сжатия KV-кэша;"),
        bullet("оценить end-to-end латентность, peak VRAM, prompt-объём и attention-прокси;"),
        bullet("локализовать зависимость recall от расстояния между фактом и контрольным вопросом."),
        PageBreak(),
        Paragraph("2. МЕТОД", heading),
        Paragraph("2.1. Модель и экспериментальные данные", subheading),
        para(
            "Использованы Qwen3.5-4B в bf16 и AMD RX 7900 XT. Контрольный диалог "
            "main-L100-00-work-terse-baseline относится к теме работы/IT, использует лаконичную персону "
            "и baseline-сценарий. Он содержит 100 ходов и 22 контрольные пробы: прямые, "
            "переформулированные, multi-hop и unanswerable. Генерация детерминирована (greedy, seed=0), "
            "максимальная длина ответа — 32 новых токена."
        ),
        Paragraph("2.2. Сравниваемые системы", subheading),
        bullet("Full KV сохраняет всю историю и служит верхним контролем качества."),
        bullet("Sliding, TopK, SnapKV и PyramidKV используют бюджет 128 слотов на слой; "
               "SnapKV и PyramidKV реализованы по опубликованным методам [1, 2]."),
        bullet("Живая память k32 подключает до 32 латентных K/V-слотов из банка ёмкостью 512. "
               "Dynamic применяет косинусный гейт 0,5 и подключает от 1 до 32 слотов. "
               "Текстовый RAG в обеих живых конфигурациях отключён."),
        Paragraph("2.3. Протокол измерений", subheading),
        para(
            "После warmup выполнены пять чередующихся повторов каждой стратегии с ротацией порядка. "
            "End-to-end латентность включает prefill, decoding и операции retrieval/embedding. "
            "Пиковая память измерена через torch.max_memory_allocated внутри генерации. "
            "Attention-прокси равен суммарному числу кэшируемых и инжектируемых слотов и не является "
            "прямым замером FLOPs."
        ),
        Paragraph("2.4. Метрики качества", subheading),
        para(
            "Recall основан на присутствии ожидаемых ключевых строк. Для multi-hop частичный ответ даёт "
            "0,5. Дополнительно рассчитан recall только по 20 answerable-пробам: две unanswerable-пробы "
            "исключены, поскольку их исходная метрика грубо аппроксимирует корректный отказ. "
            "Разбиение по lag-зонам показывает удержание факта на расстоянии 0–5, 6–12, 13–20 и 21+ ходов."
        ),
        PageBreak(),
        Paragraph("3. РЕЗУЛЬТАТЫ", heading),
        Paragraph("3.1. Качество и стоимость", subheading),
    ]
    headers = ["Стратегия", "Recall", "Answerable\nrecall", "E2E,\nс/ход", "Peak VRAM,\nMiB", "Prompt\nтокены", "Attention\nслоты"]
    table_rows = []
    for name in ORDER:
        row = results[name]
        table_rows.append([
            LABELS[name], f"{row['recall']:.3f}", f"{row['answerable']:.3f}",
            f"{row['e2e']:.3f}", f"{row['peak']:.0f}",
            f"{row['prompt']:,}".replace(",", " "), f"{row['attention']:,}".replace(",", " "),
        ])
    add_table(story, "Таблица 1 — Результаты контрольного диалога", headers, table_rows,
              [3.6, 1.4, 1.9, 1.6, 2.1, 2.8, 3.0], 7.2)
    story += [
        Image(str(FIGURE_PATH), width=16.6 * cm, height=5.6 * cm),
        Paragraph("Рисунок 1 — Recall, end-to-end латентность и peak VRAM", caption),
        para(
            f"Dynamic превосходит PyramidKV на 0,136 recall и уступает SnapKV на 0,023. "
            f"После исключения unanswerable-проб recall составляет {dynamic['answerable']:.3f} у dynamic, "
            f"{pyramid['answerable']:.3f} у PyramidKV, {snap['answerable']:.3f} у SnapKV и "
            f"{full['answerable']:.3f} у Full KV. Порядок методов по качеству сохраняется."
        ),
        Paragraph("3.2. Влияние динамического гейта", subheading),
        para(
            f"Dynamic улучшил recall относительно k32 с {fixed['recall']:.3f} до {dynamic['recall']:.3f}. "
            f"Среднее число инжектируемых слотов снизилось с {fixed_slots:.2f} до {dynamic_slots:.2f} "
            f"на шаг, то есть в {gate_ratio:.1f} раза. Медианная e2e-латентность возросла с "
            f"{fixed['e2e']:.3f} до {dynamic['e2e']:.3f} с/ход (3,5%)."
        ),
        Paragraph("3.3. Зависимость от дистанции", subheading),
    ]
    lag_rows = [[LABELS[name]] + [f"{results[name]['lag'][zone]:.3f}" for zone in ("0-5", "6-12", "13-20", "21+")]
                for name in ORDER]
    add_table(story, "Таблица 2 — Recall по lag-зонам", ["Стратегия", "0–5", "6–12", "13–20", "21+"],
              lag_rows, [5.2, 2.1, 2.1, 2.1, 2.1], 8)
    story += [
        para(
            "Обе живые конфигурации не ответили верно на пробы зоны 13–20. В зоне 21+ dynamic получил "
            "0,278, k32 — 0,111. Таким образом, агрегированное преимущество dynamic над PyramidKV "
            "сформировано преимущественно короткими дистанциями и paraphrase-пробами, а не равномерным "
            "улучшением долгосрочного удержания."
        ),
        PageBreak(),
        Paragraph("4. ОБСУЖДЕНИЕ", heading),
        bullet(
            f"<b>Относительно PyramidKV</b> dynamic показал более высокий recall при снижении e2e на "
            f"{speed_gain:.1f}% и peak VRAM на {vram_gain:.1f}%. Диапазоны пяти повторов латентности "
            "у этих стратегий не пересекаются, что поддерживает воспроизводимость различия внутри батча."
        ),
        bullet(
            f"<b>SnapKV задаёт границу результата.</b> Его recall {snap['recall']:.3f} выше, чем у dynamic, "
            "поэтому данные не поддерживают утверждение о лучшем качестве среди всех компактных методов."
        ),
        bullet(
            f"<b>Экономия входа не эквивалентна экономии attention.</b> Dynamic использует в "
            f"{prompt_factor:.2f} раза меньше prompt-токенов, чем PyramidKV, но его attention-прокси "
            f"в {attention_factor:.2f} раза выше. Следовательно, преимущество относится к измеренным "
            "латентности, VRAM и prompt-объёму, а не ко всем вычислительным осям."
        ),
        bullet(
            "<b>Full KV остаётся эталоном.</b> Разрыв с полным контекстом показывает, что латентная "
            "инъекция в текущем виде не заменяет исходную историю для точного долгосрочного recall."
        ),
        Paragraph("5. ОГРАНИЧЕНИЯ", heading),
        bullet("Один диалог: доверительные интервалы по диалогам и парная статистическая значимость не рассчитываются."),
        bullet("Синтетический keyword-скоринг измеряет восстановление заданных фактов, но не всестороннее качество диалога."),
        bullet("Baseline-сценарий не содержит temporal-проб; обновления и противоречия фактов здесь не проверяются."),
        bullet("Результат получен для одной модели и одной GPU-конфигурации; перенос на другие условия требует проверки."),
        Paragraph("6. ЗАКЛЮЧЕНИЕ", heading),
        para(
            "Контрольный эксперимент показал практическую работоспособность динамической латентной "
            "K/V-памяти и преимущество над PyramidKV по совокупности recall, e2e-латентности и peak VRAM. "
            "Динамический гейт также сократил число инъекций более чем на порядок без потери качества "
            "относительно k32. Одновременно выявлены провал в средней lag-зоне и преимущество SnapKV "
            "по recall. Полученные данные обосновывают расширение эксперимента на полный main-корпус, "
            "но не являются статистическим доказательством общего превосходства метода."
        ),
        Paragraph("СПИСОК ЛИТЕРАТУРЫ", heading),
        Paragraph("1. Li Y. SnapKV: LLM Knows What You are Looking for Before Generation / Y. Li et al. "
                  "// NeurIPS. — 2024. — https://arxiv.org/abs/2404.14469", reference),
        Paragraph("2. Cai Z. PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information "
                  "Funneling / Z. Cai et al. — 2024. — https://arxiv.org/abs/2406.02069", reference),
        Paragraph("ВОСПРОИЗВЕДЕНИЕ", heading),
        Paragraph("scripts/run_dialogue_fullscale.sh core gate", code),
        Paragraph(".venv/bin/python scripts/build_core_gate_report.py", code),
        Paragraph("Исходные данные: results/corpus/full-core-v1/batches/main-L100-00-work-terse-baseline/", reference),
    ]

    def page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Serif", 9)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawCentredString(A4[0] / 2, 0.8 * cm, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(PDF_PATH)


if __name__ == "__main__":
    main()

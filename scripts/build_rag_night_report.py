"""Build a scientific PDF explaining the full_rag architectures and night results."""

from __future__ import annotations

import html
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results/corpus/full-rag-v1"
BATCHES = [
    "main-L100-00-work-terse-baseline",
    "main-L100-01-travel-chatty-updates",
]
PDF_PATH = RUN_DIR / "RAG_NIGHT_REPORT.pdf"
PAIR_FIGURE = RUN_DIR / "rag_pair_recall.png"
DELTA_FIGURE = RUN_DIR / "rag_pair_delta_by_dialogue.png"
PAIRS = [
    ("last, raw", "rag_text_only", "v4_rag_kv_plain"),
    ("last, centered", "rag_last_centered", "v4_rag_kv"),
    ("mid, raw", "rag_latent_mid", "v4_rag_kv_mid_plain"),
    ("mid, centered", "rag_mid_centered", "v4_rag_kv_mid"),
    ("mean, raw", "rag_mean", "v4_rag_kv_mean"),
    ("mean, centered", "rag_mean_centered", "v4_rag_kv_mean_centered"),
]
LABELS = {
    "classic_kv_full": "Full KV",
    "rag_text_only": "rag_text_only",
    "rag_last_centered": "rag_last_centered",
    "rag_latent_mid": "rag_latent_mid",
    "rag_mid_centered": "rag_mid_centered",
    "rag_mean": "rag_mean",
    "rag_mean_centered": "rag_mean_centered",
    "rag_bank_key": "rag_bank_key",
    "rag_bank_dev": "rag_bank_dev",
    "v4_rag_kv_plain": "v4_rag_kv_plain",
    "v4_rag_kv": "v4_rag_kv",
    "v4_rag_kv_mid_plain": "v4_rag_kv_mid_plain",
    "v4_rag_kv_mid": "v4_rag_kv_mid",
    "v4_rag_kv_mean": "v4_rag_kv_mean",
    "v4_rag_kv_mean_centered": "v4_rag_kv_mean_centered",
}
NAME_ROWS = [
    ("classic_kv_full", "Вся история", "—", "Нет"),
    ("rag_text_only", "last", "raw", "Нет"),
    ("rag_last_centered", "last", "centered", "Нет"),
    ("rag_latent_mid", "mid", "raw", "Нет"),
    ("rag_mid_centered", "mid", "centered", "Нет"),
    ("rag_mean", "mean", "raw", "Нет"),
    ("rag_mean_centered", "mean", "centered", "Нет"),
    ("rag_bank_key", "Слоты банка", "raw", "Нет"),
    ("rag_bank_dev", "Слоты банка", "anchor/dev", "Нет"),
    ("v4_rag_kv_plain", "last", "raw", "Да, k32"),
    ("v4_rag_kv", "last", "centered", "Да, k32"),
    ("v4_rag_kv_mid_plain", "mid", "raw", "Да, k32"),
    ("v4_rag_kv_mid", "mid", "centered", "Да, k32"),
    ("v4_rag_kv_mean", "mean", "raw", "Да, k32"),
    ("v4_rag_kv_mean_centered", "mean", "centered", "Да, k32"),
]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    data = {}
    for batch_id in BATCHES:
        data[batch_id] = {}
        for path in (RUN_DIR / "batches" / batch_id).glob("*.json"):
            payload = json.loads(path.read_text())
            analysis = payload["analysis"]
            answerable = [p["score"] or 0.0 for p in analysis["probe_details"]
                          if p["kind"] != "unanswerable"]
            data[batch_id][payload["strategy"]] = {
                "recall": analysis["recall_mean"],
                "answerable": mean(answerable),
                "e2e": median(payload["e2e"]["per_rep_mean_s"]),
                "peak": analysis["peak_mem_mb"],
                "prompt": analysis["prompt_tokens"],
                "temporal": analysis["recall_by_kind"].get("temporal"),
                "stale_updated": analysis["stale_updated_rate"],
            }

    def avg(name, key):
        return round(mean(data[b][name][key] for b in BATCHES), 4)

    pair_rows = []
    for label, pure, dual in PAIRS:
        pure_raw = mean(data[b][pure]["recall"] for b in BATCHES)
        dual_raw = mean(data[b][dual]["recall"] for b in BATCHES)
        pure_avg = round(pure_raw, 4)
        dual_avg = round(dual_raw, 4)
        deltas = [data[b][dual]["recall"] - data[b][pure]["recall"] for b in BATCHES]
        pure_e2e = avg(pure, "e2e")
        dual_e2e = avg(dual, "e2e")
        pair_rows.append({
            "label": label, "pure": pure, "dual": dual,
            "pure_avg": pure_avg, "dual_avg": dual_avg,
            "delta": dual_raw - pure_raw, "deltas": deltas,
            "e2e_delta": (dual_e2e / pure_e2e - 1) * 100,
        })

    full = avg("classic_kv_full", "recall")
    best_hybrid = max(pair_rows, key=lambda row: row["dual_avg"])
    best_pure = max(pair_rows, key=lambda row: row["pure_avg"])
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    x = np.arange(len(PAIRS))
    width = 0.36
    pure_values = [row["pure_avg"] for row in pair_rows]
    dual_values = [row["dual_avg"] for row in pair_rows]
    bars_pure = ax.bar(x - width / 2, pure_values, width, label="RAG-only", color="#64748b")
    bars_dual = ax.bar(x + width / 2, dual_values, width, label="RAG + живая K/V", color="#059669")
    ax.axhline(full, color="#b91c1c", linestyle="--", linewidth=1.4, label=f"Full KV = {full:.3f}")
    ax.set_xticks(x, [row["label"] for row in pair_rows], rotation=20, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Recall, среднее по двум диалогам")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    for bars in (bars_pure, bars_dual):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8)
    ax.set_title("Эффект добавления живой памяти к одному и тому же RAG")
    fig.tight_layout()
    fig.savefig(PAIR_FIGURE, dpi=190, bbox_inches="tight")
    plt.close(fig)

    delta_matrix = np.array([row["deltas"] for row in pair_rows])
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    limit = max(abs(delta_matrix.min()), abs(delta_matrix.max()))
    image = ax.imshow(delta_matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks([0, 1], ["baseline", "updates"])
    ax.set_yticks(np.arange(len(PAIRS)), [row["label"] for row in pair_rows])
    for i in range(delta_matrix.shape[0]):
        for j in range(delta_matrix.shape[1]):
            value = delta_matrix[i, j]
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center",
                    color="black", fontsize=10, fontweight="bold")
    ax.set_title("Δ recall: живая память минус тот же RAG без инъекции")
    fig.colorbar(image, ax=ax, label="Δ recall")
    fig.tight_layout()
    fig.savefig(DELTA_FIGURE, dpi=190, bbox_inches="tight")
    plt.close(fig)

    font_dir = Path("/usr/share/fonts/truetype/liberation")
    pdfmetrics.registerFont(TTFont("Serif", str(font_dir / "LiberationSerif-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("Serif-Bold", str(font_dir / "LiberationSerif-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Serif-Italic", str(font_dir / "LiberationSerif-Italic.ttf")))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic")
    mono_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    pdfmetrics.registerFont(TTFont("Mono", str(mono_dir / "DejaVuSansMono.ttf")))
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontName="Serif-Bold",
                           fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=10)
    subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName="Serif-Italic",
                              fontSize=10, leading=13, alignment=TA_CENTER,
                              textColor=colors.HexColor("#475569"), spaceAfter=12)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontName="Serif",
                          fontSize=10.4, leading=14.2, alignment=TA_JUSTIFY,
                          firstLineIndent=0.7 * cm, spaceAfter=5, splitLongWords=True)
    no_indent = ParagraphStyle("NoIndent", parent=body, firstLineIndent=0)
    heading = ParagraphStyle("Heading", parent=body, fontName="Serif-Bold", fontSize=13,
                             leading=16, alignment=TA_LEFT, firstLineIndent=0,
                             spaceBefore=9, spaceAfter=6, keepWithNext=True)
    subheading = ParagraphStyle("Subheading", parent=heading, fontSize=11,
                                leading=14, spaceBefore=6, spaceAfter=4)
    caption = ParagraphStyle("Caption", parent=no_indent, fontSize=9.3, leading=12,
                             alignment=TA_CENTER, spaceBefore=4, spaceAfter=7)
    table_text = ParagraphStyle("TableText", parent=no_indent, fontSize=7.4, leading=9,
                                alignment=TA_CENTER)
    table_head = ParagraphStyle("TableHead", parent=table_text, fontName="Serif-Bold")
    name_text = ParagraphStyle("NameText", parent=table_text, fontName="Mono", fontSize=6.8,
                               leading=8.2, alignment=TA_LEFT)
    reference = ParagraphStyle("Reference", parent=no_indent, fontSize=9, leading=12,
                               leftIndent=0.6 * cm, firstLineIndent=-0.6 * cm, spaceAfter=4)
    code = ParagraphStyle("Code", parent=no_indent, fontName="Mono", fontSize=8.2,
                          leading=11, leftIndent=0.4 * cm, spaceAfter=3)

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4,
                            leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="Текстовый RAG и живая K/V-память: архитектуры и результаты",
                            author="Transformer experiments",
                            subject="Пояснение архитектур и анализ full-rag-v1")

    def para(text, style=body):
        return Paragraph(text, style)

    def bullet(text):
        return Paragraph(f"• {text}", no_indent)

    def add_table(story, table_caption, headers, rows, widths, font_size=7.4, names=False):
        story.append(Paragraph(html.escape(table_caption), caption))
        cell_style = ParagraphStyle("SizedCell", parent=table_text, fontSize=font_size,
                                    leading=font_size * 1.2)
        head_style = ParagraphStyle("SizedHead", parent=table_head, fontSize=font_size,
                                    leading=font_size * 1.2)
        data_rows = [[Paragraph(html.escape(str(cell)), head_style) for cell in headers]]
        for row in rows:
            cells = []
            for index, cell in enumerate(row):
                style = ParagraphStyle("NameCell", parent=name_text, fontSize=font_size * 0.92,
                                       leading=font_size * 1.1) if names and index == 0 else cell_style
                cells.append(Paragraph(html.escape(str(cell)), style))
            data_rows.append(cells)
        table = Table(data_rows, colWidths=[w * cm for w in widths], repeatRows=1, hAlign="CENTER")
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#64748b")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story += [table, Spacer(1, 0.18 * cm)]

    story = [
        Paragraph("Текстовый RAG и живая K/V-память: архитектуры, термины и "
                  "результаты совмещения", title),
        Paragraph("Научный отчёт · full-rag-v1 · два диалога · 18 ноября 2026", subtitle),
        Paragraph("АННОТАЦИЯ", heading),
        para(
            f"Проведено сравнение текстового retrieval и гибридной архитектуры, в которой к текстовым "
            f"заметкам добавляется инъекция латентных K/V. На двух диалогах лучшие гибриды — mid raw "
            f"и mean raw — получили recall {pair_rows[2]['dual_avg']:.3f} и {pair_rows[4]['dual_avg']:.3f} "
            f"против {best_pure['pure_avg']:.3f} у лучшего чистого RAG. Прирост в соответствующих парах "
            f"составил +0,091 и +0,072. Full KV остался лучшим ({full:.3f}). Центрированные гибриды не "
            "показали общего преимущества. Результат предварительный: число диалогов мало, а сценарии "
            "конфаундированы темой и персоной."
        ),
        para("<b>Ключевые слова:</b> retrieval-augmented generation; латентная память; K/V-инъекция; "
             "центрирование; средние слои; диалоговый recall.", no_indent),
        Paragraph("1. ЦЕЛЬ ОТЧЁТА", heading),
        para(
            "Отчёт решает две задачи. Во-первых, фиксирует точный смысл названий стратегий, чтобы "
            "last/mid/mean, raw/centered и bank/bank_dev не смешивались между собой. Во-вторых, "
            "оценивает эффект добавления живой памяти к одному и тому же RAG: каждая гибридная "
            "конфигурация сравнивается прежде всего со своей парой без K/V-инъекции."
        ),
        Paragraph("Исследовательский вопрос", subheading),
        para(
            "Улучшает ли латентная K/V-инъекция текстовый retrieval, и зависит ли эффект от слоя "
            "формирования ключей, центрирования и сценария диалога? Отдельно проверяется цена гибрида "
            "по end-to-end латентности, temporal recall и частоте устаревших ответов."
        ),
        PageBreak(),
        Paragraph("2. ЧТО ЕСТЬ ЧТО: АРХИТЕКТУРЫ И ТЕРМИНЫ", heading),
        Paragraph("2.1. Текстовый RAG", subheading),
        para(
            "RAG-банк хранит <b>тексты реплик</b> и их векторные ключи. Для текущего вопроса выбираются "
            "наиболее близкие по косинусному сходству заметки, и их исходный текст добавляется в промпт. "
            "Стратегии с префиксом <font name='Mono'>rag_</font> используют только этот текстовый канал: "
            "K/V-инъекции в attention у них нет."
        ),
        Paragraph("2.2. Живая память", subheading),
        para(
            "Живая память — банк латентных активаций средних full-attention слоёв. Его векторы проецируются "
            "в дополнительные ключи и значения и подключаются к attention при генерации. В чистом виде "
            "текст в промпт не возвращается. В гибридах <font name='Mono'>v4_rag_kv_*</font> этот канал "
            "работает <b>вместе</b> с текстовым RAG. В данном прогоне все гибриды используют фиксированные "
            "k32; динамический гейт из core-эксперимента здесь не применяется."
        ),
        Paragraph("2.3. Слой формирования ключей", subheading),
        bullet("<b>last</b> — среднее по токенам скрытых состояний последнего слоя <font name='Mono'>hs[-1]</font>."),
        bullet("<b>mid</b> — среднее по токенам одного среднего слоя <font name='Mono'>hs[len(hs)//2]</font>."),
        bullet("<b>mean</b> — сначала усреднение всех слоёв кроме embedding (<font name='Mono'>hs[1:]</font>), "
               "затем среднее по токенам. Это не один средний слой: mid и mean — разные представления."),
        Paragraph("2.4. Raw и centered", subheading),
        bullet("<b>raw</b> — исходный эмбеддинг нормализуется без вычитания центроида."),
        bullet("<b>centered</b> — из эмбеддинга вычитается running mean ранее сохранённых RAG-эмбеддингов, "
               "после чего выполняется нормализация. Это глобальное центрирование всего текстового банка."),
        Paragraph("2.5. Bank и bank_dev", subheading),
        bullet("<b>rag_bank_key</b> — ключ реплики строится как softmax-взвешенная смесь до 32 ближайших "
               "слотов живого банка. Слоты используются для поиска текста, но не инжектятся в attention."),
        bullet("<b>rag_bank_dev</b> — top-1 слот становится якорем кластера; из ключа вычитается якорь, "
               "а поиск внутри кластера идёт по отклонению. Это кластерное anchor/deviation-кодирование, "
               "а не глобальный centered из раздела 2.4. K/V-инъекции также нет."),
        PageBreak(),
        Paragraph("3. РАСШИФРОВКА ВСЕХ НАЗВАНИЙ", heading),
        para(
            "Колонка «K/V» отвечает на главный вопрос: добавляется ли живая память к текстовому RAG. "
            "Одинаковые слой и центрирование при разных значениях этой колонки образуют контрольную пару."
        ),
    ]
    add_table(story, "Таблица 1 — Состав стратегий full-rag-v1",
              ["Имя стратегии", "Источник ключей", "Центрирование", "Живая K/V"],
              NAME_ROWS, [6.3, 3.5, 3.0, 2.7], 7.3, names=True)
    story += [
        Paragraph("Примеры чтения названий", subheading),
        bullet("<font name='Mono'>rag_mid_centered</font>: чистый текстовый RAG; ключи одного среднего слоя; "
               "глобальный центроид; живой K/V-инъекции нет."),
        bullet("<font name='Mono'>v4_rag_kv_mid</font>: тот же mid-centered RAG <b>плюс</b> живая K/V-память. "
               "Сравнение с предыдущей стратегией измеряет именно эффект добавления живого канала."),
        bullet("<font name='Mono'>v4_rag_kv_mean</font>: текстовый RAG с усреднением всех слоёв без "
               "центрирования плюс фиксированная K/V-инъекция k32."),
        bullet("<font name='Mono'>rag_bank_dev</font>: особый случай — текстовый поиск по кластерным "
               "отклонениям слотов живого банка, но без добавления K/V в attention."),
        PageBreak(),
        Paragraph("4. ПРОТОКОЛ И ОСНОВНОЙ РЕЗУЛЬТАТ", heading),
        para(
            "Использованы Qwen3.5-4B bf16, AMD RX 7900 XT, greedy и до 32 новых токенов. Каждый из двух "
            "диалогов содержит 100 ходов; первый — work/terse/baseline (22 пробы), второй — "
            "travel/chatty/updates (24 пробы). После warmup выполнены пять чередующихся повторов. "
            "E2E включает prefill, decoding, retrieval и embedding. Общее GPU-время — 6,89 часа; "
            "оба батча завершены без ошибок."
        ),
    ]
    result_rows = [[
        row["label"], f"{row['pure_avg']:.3f}", f"{row['dual_avg']:.3f}",
        f"{row['delta']:+.3f}", f"{row['deltas'][0]:+.3f}", f"{row['deltas'][1]:+.3f}",
        f"{row['e2e_delta']:+.1f}%",
    ] for row in pair_rows]
    add_table(story, "Таблица 2 — Парное сравнение RAG-only и RAG + живая память",
              ["Ключи", "RAG-only", "+K/V", "Δ среднее", "Δ baseline", "Δ updates", "Δ E2E"],
              result_rows, [3.2, 1.8, 1.6, 1.8, 1.9, 1.8, 1.5], 7.2)
    story += [
        Image(str(PAIR_FIGURE), width=16.7 * cm, height=7.15 * cm),
        Paragraph("Рисунок 1 — Средний recall по двум диалогам; пунктир — Full KV", caption),
        para(
            f"Наилучшие гибриды mid raw и mean raw превосходят лучший чистый RAG на 0,073 и 0,072. "
            f"При этом добавление K/V увеличивает латентность на 12–17%. Центрированные last и mean "
            "гибриды уступают своим парам без инъекции, поэтому общий вывод о пользе центрирования "
            "или живой памяти для каждой конфигурации не делается."
        ),
        PageBreak(),
        Paragraph("5. РАЗНОСЦЕНАРНОСТЬ И ОБНОВЛЕНИЯ ФАКТОВ", heading),
        Image(str(DELTA_FIGURE), width=14.8 * cm, height=9.1 * cm),
        Paragraph("Рисунок 2 — Изменение recall от добавления K/V на каждом диалоге", caption),
        para(
            "mid raw даёт +0,181 на baseline и нулевой эффект на updates; mean raw даёт −0,023 на baseline "
            "и +0,167 на updates. Следовательно, конфигурации не образуют единый устойчивый порядок на "
            "обоих диалогах. Поскольку тема, персона и seed различаются вместе со сценарием, рисунок "
            "показывает междиалоговую вариативность, а не чистый причинный эффект updates."
        ),
    ]
    selected = [
        "classic_kv_full", "rag_mid_centered", "rag_mean", "v4_rag_kv_mid_plain",
        "v4_rag_kv_mean", "v4_rag_kv_mid", "v4_rag_kv_mean_centered", "rag_bank_dev",
    ]
    detail_rows = []
    for name in selected:
        temporal = data[BATCHES[1]][name]["temporal"]
        stale = data[BATCHES[1]][name]["stale_updated"]
        detail_rows.append([
            LABELS[name], f"{avg(name, 'answerable'):.3f}",
            "—" if temporal is None else f"{temporal:.3f}",
            "—" if stale is None else f"{stale:.3f}",
            f"{avg(name, 'e2e'):.3f}",
            f"{max(data[b][name]['peak'] for b in BATCHES):.0f}",
        ])
    add_table(story, "Таблица 3 — Answerable recall, updates-метрики и стоимость",
              ["Стратегия", "Answerable", "Temporal\nupdates", "Stale updated\n↓", "E2E,\nс/ход", "Peak max,\nMiB"],
              detail_rows, [5.1, 2.0, 2.0, 2.2, 1.8, 2.0], 7.0, names=True)
    story += [
        para(
            "Stale updated — доля ответов со старым значением среди обновлённых фактов; меньше лучше. "
            "У лучших raw-гибридов temporal=0,5 и stale=0,25; у centered-гибридов — 0,0 и 0,5. "
            "Full KV достигает temporal=1,0. Поэтому преимущество raw-гибридов по общему recall не "
            "устраняет проблему обновлений и противоречий."
        ),
        PageBreak(),
        Paragraph("6. ОБСУЖДЕНИЕ И ЗАКЛЮЧЕНИЕ", heading),
        bullet(
            f"<b>Гибридная архитектура дала положительный сигнал.</b> {best_hybrid['dual']} показал "
            f"recall {best_hybrid['dual_avg']:.3f}; mean-гибрид практически совпадает с ним. "
            "Это выше всех чистых текстовых RAG в данном прогоне."
        ),
        bullet(
            "<b>Raw mid/mean сейчас информативнее centered-гибридов.</b> Центрирование может быть "
            "полезным внутри чистого RAG, но его взаимодействие с K/V-инъекцией нестабильно и "
            "в ряде пар отрицательно."
        ),
        bullet(
            "<b>Банковые ключи не подтвердились.</b> rag_bank_key и rag_bank_dev получили 0,474 и 0,506 "
            "соответственно. Использование слотов живого банка как ключей не следует смешивать с "
            "успешным сочетанием обычного текстового RAG и K/V-инъекции."
        ),
        bullet(
            "<b>Full KV остаётся эталоном.</b> Разрыв по общему качеству и temporal recall показывает, "
            "что гибрид пока не заменяет исходную историю для точного удержания фактов."
        ),
        Paragraph("Ограничения", subheading),
        bullet("Только два диалога: парная статистическая значимость не установлена."),
        bullet("Сценарий конфаундирован темой, персоной и seed; различия baseline/updates нельзя интерпретировать причинно."),
        bullet("Keyword-скоринг измеряет восстановление заданных строк, а не полное качество ответа."),
        bullet("Все гибриды используют k32; перенос вывода на динамический гейт требует отдельного эксперимента."),
        Paragraph("Практический вывод", subheading),
        para(
            "В следующих прогонах приоритетными кандидатами являются mid raw + K/V и mean raw + K/V. "
            "Их следует сравнивать с одинаковыми RAG-only парами на большем числе тем и сценариев, "
            "отдельно отслеживая temporal и stale updated. Центрирование не следует удалять из плана, "
            "но его эффект нужно оценивать независимо для чистого RAG и для гибрида."
        ),
        Paragraph("ВОСПРОИЗВЕДЕНИЕ И ИСТОЧНИКИ ОПРЕДЕЛЕНИЙ", heading),
        Paragraph("scripts/run_dialogue_fullscale.sh rag night", code),
        Paragraph(".venv/bin/python scripts/build_rag_night_report.py", code),
        Paragraph("Ключи last/mid/mean: tmem/semantic.py:43–65. Bank/bank_dev: tmem/dialogue.py:245–267 "
                  "и tmem/semantic.py:90–208. Исходные результаты: results/corpus/full-rag-v1/batches/.", reference),
    ]

    def page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Serif", 9)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawCentredString(A4[0] / 2, 0.75 * cm, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(PDF_PATH)


if __name__ == "__main__":
    main()

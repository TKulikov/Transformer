"""Build the concise Russian report from the common Qwen3.5 measurements."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_data as RD


def summary(q: dict) -> str:
    ref = q["classic_kv_full"]
    final = q["v4_rag_kv"]
    factor = ref["cache_slots"] / final["cache_slots"]
    return (
        f"Чистый Qwen3.5-4B с полным контекстом ответил верно на {round(ref['recall_mean'] * 24)}/24 "
        f"контрольных вопросов. Двухканальная система получила recall {final['recall_mean']:.3f} при "
        f"{factor:.2f}× меньшей прокси-работе attention. Наблюденная средняя латентность генерации — "
        f"{ref['mean_turn_latency_s']:.2f} с у чистого Qwen и {final['mean_turn_latency_s']:.2f} с у "
        "двухканальной системы. Это результат одного полного прогона, а не доказанное устойчивое ускорение."
    )


def methodology() -> list[str]:
    return [
        "Одна модель Qwen/Qwen3.5-4B, bf16, AMD GPU, greedy-декодирование, до 32 новых токенов, seed=0.",
        "Диалог из 107 ходов: 16 фактов, 2 смены фактов и 24 вопроса; 12 прямых и 12 переформулированных.",
        "StreamingLLM, H2O, SnapKV и PyramidKV реализованы в проекте по статьям, а не взяты из официальных репозиториев. Для KV-методов средний бюджет — 128 слотов на full-attention слой; StreamingLLM хранит 4 sink + 124 recent, H2O — 4 sink + 64 recent + до 60 heavy hitters.",
        "Текстовый RAG-контроль — буфер 10 последних ходов (10 пар «реплика — ответ», window = 5) и обычная семантическая выдача текстовых заметок без центроида, скрытых тем и attention-инъекции. Это MemGPT-подобный контроль, но не полный MemGPT.",
        "Двухканальная система добавляет к текстовому retrieval латентный канал: активации скрытых слоёв извлекаются по косинусной близости, проецируются в дополнительные K/V и подключаются к четырём средним full-attention слоям без преобразования в текст.",
        "Memorizing Transformer, RMT, Landmark Attention и Memory Layers требуют специального обучения или архитектуры; они остаются в обзоре литературы и не получают выдуманных чисел в этой батарее.",
    ]


def caveats() -> list[str]:
    return [
        "Recall — keyword-скоринг; тон — косинусная близость ответа к репликам диалога, а не независимая человеческая оценка.",
        "Stale показывает любое упоминание прежнего значения. Ответ может одновременно назвать новое верное значение и старое как прежнее; такой ответ не обязательно фактически ошибочен. Знаменатель stale — все 24 вопроса.",
        "Attention-слоты — сумма средних длин KV-кэша по шагам генерации, сопоставимый прокси работы, но не прямой замер FLOPs.",
        "Латентность включает prefill и decoding, но не внешние embedding/retrieval. Peak MiB относится к вызовам генерации. Один прогон без interleaved-повторов не поддерживает сильных выводов о скорости.",
        "n=24, один seed: различия в один вопрос равны 0.042 recall; статистическая значимость не оценивалась.",
    ]


def markdown(q: dict) -> str:
    lines = [
        "# Чистое сравнение памяти и KV-кэша на Qwen3.5-4B", "",
        "## Результат", "", summary(q), "", "## Качество", "",
        "| " + " | ".join(RD.QUALITY_HEADERS) + " |", "|" + "---|" * len(RD.QUALITY_HEADERS),
    ]
    lines += ["| " + " | ".join(row) + " |" for row in RD.quality_rows(q)]
    lines += ["", "## Стоимость", "", "| " + " | ".join(RD.COST_HEADERS) + " |", "|" + "---|" * len(RD.COST_HEADERS)]
    lines += ["| " + " | ".join(row) + " |" for row in RD.cost_rows(q)]
    lines += ["", "![Сравнение](report/figs/comparison.png)", "", "## Протокол", ""]
    lines += [f"- {item}" for item in methodology()]
    lines += ["", "## Ограничения", ""] + [f"- {item}" for item in caveats()]
    lines += [
        "", "## Источники методов", "",
        "- StreamingLLM: https://arxiv.org/abs/2309.17453",
        "- H2O: https://arxiv.org/abs/2306.14048",
        "- SnapKV: https://arxiv.org/abs/2404.14469",
        "- PyramidKV: https://arxiv.org/abs/2406.02069",
        "- MemGPT (только концептуальный ориентир): https://arxiv.org/abs/2310.08560",
        "", "## Воспроизведение", "", "```bash",
        "HF_HUB_OFFLINE=1 .venv/bin/python -m tmem.cli --model unused --battery100 --models qwen35-4b --device cuda --dtype bfloat16 --max-new-tokens 32 --strategies classic_kv_full sparse_sliding_128 sparse_topk_128 snapkv_128 pyramidkv_128 rag_text_only v4_rag_kv --out results/qwen35_comparison",
        ".venv/bin/python scripts/build_report.py", ".venv/bin/python scripts/build_paper.py", "```",
    ]
    return "\n".join(lines) + "\n"


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
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="DejaVu", fontSize=8.5, leading=11.5)
    heading = ParagraphStyle("heading", parent=styles["Heading2"], fontName="DejaVu-Bold", fontSize=12)
    small = ParagraphStyle("small", parent=body, fontSize=6.8, leading=8.5)
    doc = SimpleDocTemplate(str(RD.OUT / "Memory_System_Report.pdf"), pagesize=landscape(A4),
                            leftMargin=1.1 * cm, rightMargin=1.1 * cm, topMargin=1.1 * cm, bottomMargin=1.1 * cm)
    story = [Paragraph("Чистое сравнение памяти и KV-кэша на Qwen3.5-4B", styles["Title"]),
             Paragraph("Результат", heading), Paragraph(summary(q), body),
             Paragraph("Качество", heading)]
    for headers, rows in ((RD.QUALITY_HEADERS, RD.quality_rows(q)), (RD.COST_HEADERS, RD.cost_rows(q))):
        data = [[Paragraph(h, small) for h in headers]] + [[Paragraph(c, small) for c in row] for row in rows]
        widths = [5.0 * cm] + [(27.0 - 5.0) / (len(headers) - 1) * cm] * (len(headers) - 1)
        table = Table(data, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                   ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(.96, .97, 1)])]))
        story += [table, Spacer(1, .25 * cm), Paragraph("Стоимость", heading)] if headers == RD.QUALITY_HEADERS else [table]
    for title, items in (("Протокол", methodology()), ("Ограничения", caveats())):
        story.append(Paragraph(title, heading))
        story += [Paragraph(f"• {item}", body) for item in items]
    story += [Image(str(RD.FIGS / "comparison.png"), width=24.5 * cm, height=15.1 * cm)]
    doc.build(story)


def main():
    q = RD.q35_all()
    missing = set(RD.ORDER) - set(q)
    if missing:
        raise SystemExit(f"не хватает результатов: {sorted(missing)}")
    import paper_figures
    paper_figures.make_all()
    text = markdown(q)
    (RD.OUT / "Memory_System_Report.md").write_text(text.replace("](report/figs/", "](figs/"))
    (RD.RES / "FINAL_REPORT.md").write_text(text)
    build_pdf(q)
    print("report:", RD.OUT / "Memory_System_Report.pdf")


if __name__ == "__main__":
    main()

"""Build the full project report (Russian): markdown + PDF with figures.

Run:  .venv/bin/python scripts/build_report.py
Out:  results/report/Memory_System_Report.md / .pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_data as RD  # noqa: E402

FIGS = RD.FIGS
OUT = RD.OUT

ORDER = [
    ("no_cache_full", "Пересчёт без кэша (базовая корректность)"),
    ("classic_kv_full", "Эталон: весь диалог в контексте + классический KV-кэш"),
    ("sparse_sliding_128", "Разреженный KV-кэш: скользящее окно (128 слотов)"),
    ("sparse_sliding_32", "Разреженный KV-кэш: скользящее окно (32 слота)"),
    ("sparse_topk_128", "Разреженный KV-кэш: топ-K по вниманию/H2O (128)"),
    ("window2_only", "Буфер 2 хода, никакой памяти"),
    ("window5_only", "Буфер 5 ходов, никакой памяти"),
    ("v1_memory", "Ступень 1 — темы+аддитивный хук (как в ноутбуке)"),
    ("v2_memory_ema", "Ступень 1' — EMA-слияние .95/k.25 (чинит банк тем)"),
    ("v2_rag_plain", "Ступень 2 — банк фактов текстом (до фиксов)"),
    ("v2_rag_plain_fixes", "Ступень 2+ — банк текстом (после фиксов)"),
    ("v2_rag_gist_fixes", "Ступень 2+gist — банк + темы для выдачи"),
    ("v3_rag_centroid", "Ступень 3 — банк с центрированием по теме"),
    ("v3_centroid_plus_gist", "Ступень 3 — центроид + темы из ступени 1"),
    ("v3w5_rag_centroid", "Ступень 3 — центроид, буфер 5"),
    ("v4_kv_mid", "Ступень 4 — память только в attention (без текста)"),
    ("v4_rag_kv", "Ступень 4 — банк + память в attention (финальная)"),
]

GLOSS = [
    ("Точность фактов (recall)", "доля контрольных вопросов, в ответе на которые модель назвала верное заложенное значение (поиск эталонных слов в ответе)"),
    ("Удержание темы (tone, ×эталона)", "косинусная близость ответа к высказываниям этого же диалога, нормированная на значение эталона (1.00 = как полный контекст)"),
    ("Устаревшие ответы (stale)", "доля вопросов, где после смены факта модель вернула прежнее значение (меньше — лучше)"),
    ("Деградация (repetition)", "доля ответов с повторяющимися слово-трёхграммами («I I I…» и т.п.)"),
    ("Prompt-токены", "суммарные токены, поданные на вход за всю сессию"),
    ("attention-работа (слоты)", "Σ по шагам длины KV-кэша; пропорциональна FLOPs внимания — главный показатель стоимости"),
    ("×дешевле эталона", "отношение attention-работы эталона к конфигурации"),
    ("Латентность", "среднее время одного хода диалога, секунды"),
]

METHOD = [
    "Модели: Qwen2-1.5B, GPT-2, T5-base (базовые прогоны) и Qwen3.5-4B (основная батарея: гибридная архитектура — 24 слоя линейного внимания Gated DeltaNet + 8 слоёв обычного внимания).",
    "Сессии: программируемые диалоги 35–107 ходов; 9–16 фактов закладываются в начале/середине/конце, 2–3 факта по ходу меняются, 8–24 контрольных вопроса (каждый второй — переформулированный) задаются далеко от факта (задержка 30–88 ходов).",
    "Протокол: greedy-генерация, 24–32 токена ответа, seed=0; единый явный цикл декодирования с ручным управлением KV-кэшем (корректность классического кэша доказана тестом «с кэшем == без кэша, токен-в-токен», включая настоящего GPT-2).",
    "Оборудование: AMD Radeon RX 7900 XT (20 GiB), PyTorch 2.9.1+ROCm 6.3, bf16 (Qwen3.5) / fp32 (остальные).",
    "Программный стек: пакет tmem (память, хуки, политики кэша, банк фактов, батареи), 139 офлайн-тестов + integration + gpu.",
]

STAGES = [
    ("Эталон и политики кэша", "Полный контекст с классическим KV-кэшем — верхняя планка качества; пересчёт без кэша — проверка корректности; скользящий и top-k(H2O) KV-кэши — «дешёвое внимание» из литературы."),
    ("Ступень 1 — скрытые темы (как в ноутбуке)", "Банк слотов из скрытых состояний токенов: косинус-выдача, TTL, слияние sim≥0.8 усреднением; инжекция средним вектором в активации. Идея: состояние диалога вместо истории."),
    ("Ступень 1' — починка слияния", "EMA-слияние a←(1−k)a+k·b и порог 0.95: банк перестает схлопываться (51→99 отдельных фактов), но аддитивная инжекция всё ещё не читается моделью."),
    ("Ступень 2 — банк фактов текстом (RAG)", "Факты хранятся текстом, эмбеддинг = mean-pool последних скрытых состояний самой модели; релевантные заметки попадают в промпт. Две фикс-итерации: вопросы не перетирают факты; в банк пишутся только декларации (не пересказы-ответы)."),
    ("Ступень 3 — центроид темы", "Хранение и сравнение ОТКЛОНЕНИЙ от бегущего среднего банка: тема диалога становится системой координат бесплатно (без хуков), анизотропия эмбеддингов вычитается, upsert перестаёт поедать банк."),
    ("Ступень 4 — память в attention", "Исходный замысел: активные темы входят в средние full-attention слои как дополнительные ключи/значения без позиции, с расширением маски; в кэш не пишутся. GPU-стоимость канала ≈ 0 (32 ключа из 3000)."),
    ("Обучение читателя (эксперимент)", "LoRA на общих проекциях и отдельный считывающий путь с нулевым гейтом — учим модель читать факты из канала памяти, а не только тон."),
]


def rows_for_pdf(q):
    classic = q["classic_kv_full"]
    rows = []
    for key, label in ORDER:
        d = q.get(key)
        if not d:
            continue
        factor = classic["cache_slots"] / d["cache_slots"] if d.get("cache_slots") else None
        rows.append([
            label,
            RD.fmt(d["recall_mean"]),
            RD.fmt(RD.rel_tone(d, classic), "{:.2f}×"),
            RD.fmt(d.get("stale_rate")),
            RD.fmt(d.get("rep_mean")),
            f"{d['prompt_tokens']:,}",
            f"{d['cache_slots']:,}",
            RD.fmt(factor, "{:.1f}×"),
            f"{d['mean_turn_latency_s']:.2f}",
        ])
    return rows


HEADERS = ["Конфигурация", "Факты↑", "Тема ×эт.", "Устар.↓", "Деград.↓", "Prompt ток.", "Attn слоты", "×дешевле", "Лат. с/ход"]


def small_model_rows():
    """Basic comparison rows from gpu-era benchmarks (6 strategies)."""
    out = []
    for model, pretty in (("gpt2", "GPT-2"), ("t5-base", "T5-base"), ("qwen2-1.5b", "Qwen2-1.5B")):
        b = RD.load_bench(model)
        if not b:
            continue
        for cfg, lbl in (("classic_kv_full", "классический KV"), ("sparse_sliding", "скользящий KV"),
                         ("sparse_topk", "KV top-k H2O"), ("window_only", "буфер"),
                         ("window_hybrid_memory", "буфер+память(стр.1)")):
            d = b.get(cfg)
            if not d:
                continue
            out.append([f"{pretty}: {lbl}",
                        RD.fmt(d.get("probe_mean")),
                        f"{d.get('prompt_tokens', 0):,}",
                        f"{d.get('cache_slots', 0):,}",
                        f"{d.get('mean_turn_latency_s', 0):.3f}"])
    return out


def build_pdf(rows, small_rows):
    import matplotlib
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (SimpleDocTemplate,
                                    Paragraph, Spacer, Table, TableStyle, Image, PageBreak)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    mpl_fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(mpl_fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(mpl_fonts / "DejaVuSans-Bold.ttf")))

    styles = getSampleStyleSheet()
    ru = ParagraphStyle("ru", parent=styles["Normal"], fontName="DejaVu", fontSize=8.6, leading=11.5)
    ru_h = ParagraphStyle("ruh", parent=styles["Heading2"], fontName="DejaVu-Bold", fontSize=12, leading=15)
    title_s = ParagraphStyle("t", parent=styles["Title"], fontName="DejaVu-Bold", fontSize=17, leading=22)
    small = ParagraphStyle("s", parent=ru, fontSize=7.6, leading=9.6)

    doc = SimpleDocTemplate(str(OUT / "Memory_System_Report.pdf"), pagesize=landscape(A4),
                            leftMargin=1.2 * cm, rightMargin=1.2 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm)
    story = [Paragraph("Смысловая память для диалогового LLM: полный отчёт по ступеням 1–4", title_s),
             Paragraph("Эксперименты на Qwen3.5-4B / Qwen2-1.5B / GPT-2 / T5-base · AMD RX 7900 XT (ROCm) · октябрь 2026", styles["Normal"]),
             Spacer(1, 0.4 * cm),
             Paragraph("Резюме", ru_h),
             Paragraph(
                 "Задача: заменить длинный контекст дешёвым состоянием диалога без потери качества ответа. Итог (Qwen3.5-4B, диалог 107 ходов, 24 контрольных вопроса): финальная конфигурация «банк фактов в промпте (центрированный по теме) + темы как ключи/значения в средних attention-слоях» даёт <b>71–83% точности фактов эталона при 4–5× меньшей attention-работе</b>, удержание темы <b>выше эталона</b> (1.07×), на 30% быстрее полного контекста. Скользящий KV-кэш (литературный аналог) теряет факты полностью (0–17%). аддитивная инжекция скрытых состояний (ноутбучная версия) не приносит пользы ни в одной метрике; обучить модель «читать» память постфактум малыми средствами не удалось (оформлено как негативный результат).", ru),
             Spacer(1, 0.3 * cm),
             Paragraph("Методика", ru_h)]
    for m in METHOD:
        story.append(Paragraph(f"• {m}", ru))
    story.append(Paragraph("Глоссарий метрик", ru_h))
    g = [[Paragraph("Метрика", small), Paragraph("Что означает", small)]]
    g += [[Paragraph(a, small), Paragraph(b, small)] for a, b in GLOSS]
    t = Table(g, colWidths=[7 * cm, 21 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    story.append(t)

    story.append(PageBreak())
    story.append(Paragraph("Основная таблица: Qwen3.5-4B, батарея 107 ходов / 24 вопроса", ru_h))
    data = [[Paragraph(h, small) for h in HEADERS]] + [[Paragraph(c, small) for c in r] for r in rows]
    t = Table(data, colWidths=[9.6 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 2.3 * cm, 2.3 * cm, 1.9 * cm, 2.1 * cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.96, 0.97, 1.0)]),
        ("BACKGROUND", (0, 1), (-1, 1), colors.Color(1.0, 0.93, 0.93)),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Прим.: «Тема ×эт.» нормирована на эталон внутри своей измерительной серии; строки ранних серий («до фиксов», ступень 1, политики KV) сравниваются по этой колонке только внутри группы. Выводы опираются на серии «+фиксы», ступени 3 и 4 (общая шкала).", small))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Ступени системы", ru_h))
    st = [[Paragraph(a, small), Paragraph(b, small)] for a, b in STAGES]
    t = Table([[Paragraph("Ступень", small), Paragraph("Идея", small)]] + st, colWidths=[6.5 * cm, 22 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    story.append(t)

    story.append(PageBreak())
    story.append(Paragraph("Рис. 1. Эволюция качества и стоимости", ru_h))
    story.append(Image(str(FIGS / "evolution.png"), width=22 * cm, height=14.7 * cm))
    story.append(PageBreak())
    story.append(Paragraph("Рис. 2. Парето «точность фактов — стоимость внимания»", ru_h))
    story.append(Image(str(FIGS / "pareto.png"), width=21 * cm, height=13.7 * cm))
    story.append(PageBreak())
    story.append(Paragraph("Рис. 3. Базовое сравнение политик на малых моделях", ru_h))
    story.append(Image(str(FIGS / "smallmodels.png"), width=20 * cm, height=11.3 * cm))
    story.append(Spacer(1, 0.25 * cm))
    hdr = ["Модель: политика", "Факты", "Prompt ток.", "Attn слоты", "Лат. с/ход"]
    data = [[Paragraph(h, small) for h in hdr]] + [[Paragraph(c, small) for c in r] for r in small_rows]
    t = Table(data, colWidths=[9 * cm, 2 * cm, 3 * cm, 3 * cm, 2.6 * cm], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    story.append(t)
    story.append(PageBreak())
    story.append(Paragraph("Рис. 4. Попытки обучить «читателя» памяти (негативный результат)", ru_h))
    if (FIGS / "training.png").exists():
        story.append(Image(str(FIGS / "training.png"), width=20 * cm, height=11.4 * cm))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Выводы и рекомендации", ru_h))
    for v in [
        "Рабочая архитектура: буфер свежих ходов + банк фактов текстом (upsert-дедуп, только пользовательские декларации, отклонения от центроида темы) + опциональные темы в attention средних слоёв. Даёт 71–83% фактов эталона при 4–5× cheaper attention и тоне 1.06–1.07× эталона.",
        "Скользкий/top-k KV-кэш поверх чужой модели — плохая сделка: факты теряются лавинообразно (recall 0 при бюджете 32), экономия внимания почти не конвертируется в скорость.",
        "Аддитивная инжекция скрытых состояний (ноутбучная форма) — вредна: деградация ответов (+0.4 повторов), нулевая польза для фактов; скрытым темам место только в ранжировании выдачи и (с оговорками) в attention.",
        "Post-hoc обучение читателя памяти не взлетело при 45M обучаемых параметров за 600 шагов: канал нужен в претрейне — прямой аргумент в пользу изначального замысла «учить модель с памятью сразу».",
        "Числа получены с полным контролем корректности: классический кэш == пересчёт без кэша токен-в-токен (тесты на всех моделях), метрики воспроизводимы командами из README; 139 офлайн-тестов.",
    ]:
        story.append(Paragraph(f"• {v}", ru))
    doc.build(story)


def build_md(rows, small_rows):
    lines = ["# Смысловая память для диалогового LLM — полный отчёт (ступени 1–4)", "",
             "_Qwen3.5-4B / Qwen2-1.5B / GPT-2 / T5-base · AMD RX 7900 XT (ROCm 6.3) · октябрь 2026 · код и данные: проект `tmem`_", "",
             "## Методика", *[f"- {m}" for m in METHOD], "",
             "## Глоссарий метрик", "| Метрика | Значение |", "|---|---|", *[f"| {a} | {b} |" for a, b in GLOSS], "",
             "## Основная таблица: Qwen3.5-4B, батарея 107 ходов / 24 вопроса", "| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines += ["", "_Прим.: «Тема ×эт.» нормирована на эталон внутри своей измерительной серии; строки ранних серий («до фиксов», ступень 1, политики KV) сравниваются по этой колонке только внутри группы. Выводы опираются на серии «+фиксы», ступени 3–4 (общая шкала)._"]
    lines += ["", "## Ступени системы", "| Ступень | Идея |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in STAGES]
    lines += ["", "## Графики", "![эволюция](figs/evolution.png)", "![парето](figs/pareto.png)", "![малые модели](figs/smallmodels.png)", "![обучение читателя](figs/training.png)",
              "", "## Базовое сравнение политик на малых моделях",
              "| Модель: политика | Факты | Prompt токены | Attn слоты | Лат. с/ход |", "|---|---|---|---|---|"]
    lines += ["| " + " | ".join(r) + " |" for r in small_rows]
    lines += ["", "## Воспроизведение", "```bash",
              ".venv/bin/python -m pytest -q              # 139 офлайн-тестов",
              ".venv/bin/python -m pytest -m gpu -q        # проверки на видеокарте",
              ".venv/bin/python -m tmem.cli --model qwen2-1.5b --device cuda   # базовое сравнение",
              ".venv/bin/python -m tmem.cli --model unused --stress --models gpt2 t5-base qwen2-1.5b --device cuda",
              ".venv/bin/python -m tmem.cli --model unused --battery100 --models qwen35-4b --device cuda --dtype bfloat16",
              ".venv/bin/python -m tmem.trainreader --steps 600",
              "```"]
    (OUT / "Memory_System_Report.md").write_text("\n".join(lines))


def main():
    q = RD.q35_all()
    if not q:
        raise SystemExit("нет данных батареи qwen3.5")
    RD.fig_evolution(q)
    RD.fig_pareto(q)
    RD.fig_models()
    RD.fig_training()
    rows = rows_for_pdf(q)
    small_rows = small_model_rows()
    build_md(rows, small_rows)
    build_pdf(rows, small_rows)
    print("report:", OUT / "Memory_System_Report.pdf")


if __name__ == "__main__":
    main()

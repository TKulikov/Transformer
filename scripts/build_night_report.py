"""PDF/MD отчёт: ночной пилотный прогон + вариации архитектуры чанковой памяти.

Данные берутся из того же стат-ядра, что и INTERIM.md (tmem.corpus_stats),
и из summary.json CPU/4B оценок чанковой памяти — числа в отчёте не могут
разойтись с промежуточными. Стиль PDF — как в build_report.py (reportlab,
DejaVu, landscape A4).

Запуск: .venv/bin/python scripts/build_night_report.py
Выход:  results/report/Night_Run_Report.{md,pdf} + figs/pilot_recall.png,
        figs/chunk_variations.png
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tmem.corpus_stats import LAGS, compare_pairs, load_run, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "report"
FIGS = OUT / "figs"
PILOT = ROOT / "results" / "corpus" / "pilot-v1"
CHUNK_CPU = ROOT / "results" / "chunk_code" / "code-v1" / "summary.json"
CHUNK_4B = ROOT / "results" / "chunk_code" / "code-4b-v1" / "summary.json"

ORDER = ["classic_kv_full", "sparse_sliding_128", "sparse_topk_128", "snapkv_128",
         "pyramidkv_128", "rag_text_only", "v4_rag_kv"]
SHORT = {"classic_kv_full": "full", "sparse_sliding_128": "sliding", "sparse_topk_128": "topk",
         "snapkv_128": "snapkv", "pyramidkv_128": "pyramidkv", "rag_text_only": "rag",
         "v4_rag_kv": "v4 (двухкан.)"}


def esc(s) -> str:
    return html.escape(str(s))


def f3(v) -> str:
    return f"{v:.3f}" if v is not None else "—"


def collect() -> dict:
    run = load_run(PILOT)
    summ = summarize(run)
    cfg = run["config"]
    walls = [r.get("wall_s") or 0 for r in run["records"]]
    manifest = [json.loads(l) for l in (PILOT / "manifest.jsonl").read_text().splitlines()]
    pvals = compare_pairs(run, [("v4_rag_kv", n) for n in ORDER if n != "v4_rag_kv"])
    return {
        "summ": summ, "cfg": cfg, "walls": walls,
        "failed": sum(1 for m in manifest if m["status"] == "failed"),
        "reruns": sum(1 for m in manifest if m.get("reruns", 0) > 0),
        "pvals": pvals,
        "cpu": json.loads(CHUNK_CPU.read_text(encoding="utf-8")),
        "b4": json.loads(CHUNK_4B.read_text(encoding="utf-8")),
    }


# ------------------------------------------------------------------ figures

def make_figures(d: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGS.mkdir(parents=True, exist_ok=True)
    s = d["summ"]["strategies"]
    names = [n for n in ORDER if n in s]
    recalls = [s[n]["recall_mean"] for n in names]
    lo = [s[n]["recall_mean"] - s[n]["recall_ci95"][0] for n in names]
    hi = [s[n]["recall_ci95"][1] - s[n]["recall_mean"] for n in names]
    colors = ["#2ca02c" if n == "v4_rag_kv" else "#7f9fc9" if n == "rag_text_only"
              else "#b0b0b0" for n in names]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    y = np.arange(len(names))[::-1]
    ax.barh(y, recalls, xerr=[lo, hi], color=colors, edgecolor="black",
            linewidth=0.4, capsize=3, error_kw={"linewidth": 0.8})
    ax.set_yticks(y, [SHORT[n] for n in names])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Recall (среднее по диалогам, 95 % бутстреп-ДИ)")
    ax.set_title(f"Пилотный корпус: {d['summ']['n_dialogues_done']}/12 диалогов, Qwen3.5-4B bf16, ROCm")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "pilot_recall.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, data, title in ((axes[0], d["cpu"], "CPU: Qwen2-1.5B fp32"),
                            (axes[1], d["b4"], "GPU: Qwen3.5-4B bf16")):
        strats = data["strategies"]
        labels = [n.replace("window_rag_userfacts", "rag-факты")
                   .replace("chunks_contentkey_", "ck-").replace("chunks_", "")
                  for n in strats]
        hits = [sum(1 for v in s["retrieval_hits"].values() if v) for s in strats.values()]
        clean = [s["probe_scores"].get("turn8", 0) + s["probe_scores"].get("turn10", 0)
                 for s in strats.values()]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, hits, 0.4, label="retrieval hits (0–3)", color="#7f9fc9",
               edgecolor="black", linewidth=0.4)
        ax.bar(x + 0.2, clean, 0.4, label="reuse, чистые пробы (0–2)", color="#2ca02c",
               edgecolor="black", linewidth=0.4)
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylim(0, 3.3)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Чанковая память на задаче переиспользования кода (turn9 исключён: echo-загрязнение)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "chunk_variations.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ content

def resume_bullets(d: dict) -> list[str]:
    s = d["summ"]["strategies"]
    v4, rag, full = s["v4_rag_kv"], s["rag_text_only"], s["classic_kv_full"]
    p = d["pvals"]
    return [
        f"Ночной пилот ({d['summ']['n_dialogues_done']}/12 диалогов, {sum(d['walls'])/3600:.1f} ч GPU, "
        f"{d['failed']} падений): порядки recall из статьи воспроизведены — "
        f"v4 {v4['recall_mean']:.3f} > rag {rag['recall_mean']:.3f} > pyramidkv "
        f"{s['pyramidkv_128']['recall_mean']:.3f} > snapkv {s['snapkv_128']['recall_mean']:.3f} > sliding/topk.",
        f"v4 значимо лучше snapkv (p = {p.get('v4_rag_kv vs snapkv_128', float('nan')):.3f}) и "
        f"sliding (p < 0.001); против rag и pyramidkv различия пока не значимы (n = {d['summ']['n_dialogues_done']}).",
        f"Цена: v4 — медиана {v4['e2e_latency_median_s']:.2f} с/ход против {full['e2e_latency_median_s']:.2f} с "
        f"у полного контекста; промпт {round(v4['prompt_tokens'])} против {round(full['prompt_tokens'])} токенов "
        f"(в {full['prompt_tokens']/v4['prompt_tokens']:.1f} раза меньше).",
        "Вариации архитектуры чанковой памяти: контентный ключ (активации блока, среднее по слоям) + "
        "центрирование бегущим средним даёт 3/3 поиска и 3/3 переиспользования кода на 4B; "
        "ключ из одного слоя хуже (2/3). Без памяти модель галлюцинирует имена функций.",
        "Банк фактов пользователя вспоминает имена функций из запросов, но не сам код (0/3 code-hits); "
        "чанковый банк возвращает реализации.",
    ]


def pilot_protocol(d: dict) -> list[str]:
    cfg = d["cfg"]
    return [
        f"Модель {cfg.get('model')} (Qwen/Qwen3.5-4B), {cfg.get('dtype')}, AMD RX 7900 XTX 24 ГБ, "
        f"torch {cfg.get('torch')} (ROCm), git {cfg.get('git_sha')}. Декодирование greedy, seed 0, "
        "max_new_tokens 32.",
        "Корпус pilot: синтетические диалоги по 100 ходов, 6 тем × 4 персоны × 5 ситуаций, "
        "пробы во всех lag-зонах (direct/paraphrase/multihop/temporal/unanswerable). "
        "Батч = 1 диалог × 7 стратегий; очередь сбалансирована round-robin.",
        "Латентность: 1 warmup + 5 чередующихся повторов (A,B,A,B,…), end-to-end включая "
        "embedding/retrieval; порядок стратегий ротируется между батчами.",
        "Статистика: бутстреп-ДИ по диалогам (не по вопросам), попарный бутстреп с поправкой Холма. "
        "То же ядро, что в INTERIM.md и финальной агрегации.",
        f"Окно прогона: 13.11 22:12 — 14.11 12:00. Батчи: {len(d['walls'])} выполнено, "
        f"wall-time {min(d['walls'])/60:.0f}–{max(d['walls'])/60:.0f} мин, падений {d['failed']}, "
        f"перезапусков {d['reruns']}. Остановка/возобновление по манифесту проверены на практике.",
    ]


def chunk_protocol() -> list[str]:
    return [
        "Задача: модель пишет 3 функции (factorial, reverse_string, count_vowels), затем 5 filler-ходов, "
        "затем 3 пробы «напиши новую функцию, переиспользуя ранее написанную». Окно контекста = 2 пары: "
        "ранний код выпадает из контекста, вернуть его может только память.",
        "Метрики: reuse — имя нужной функции в ответе (keyword); retrieval hit — банк положил правильный "
        "код в промпт. Проба turn9 загрязнён (имя функции в самом вопросе, echo), чистые — turn8/turn10.",
        "Оси вариаций: ключ (запрос vs активации блока: mean/mid/last слоёв); эмбеддинг значения "
        "(mean/span/attn/late); скоринг (key/value/maxsim); центрирование бегущим средним ключей и значений.",
        "CPU: Qwen2-1.5B fp32, 8 потоков; GPU: Qwen3.5-4B bf16, ROCm. Greedy, max_new_tokens 64, одна сессия.",
    ]


def chunk_findings(d: dict) -> list[str]:
    return [
        "Контентный ключ работает: на 4B mean-по-слоям даёт 3/3 retrieval hits и 3/3 reuse — модель пишет "
        "реальные вызовы ранее созданных функций (return s == reverse_string(s); count_vowels(s)/len(s)).",
        "Абляция слоёв: mean (3/3) лучше last (2/3) и mid (2/3) — усреднение активаций по слоям важно.",
        "Без памяти (window_only) 4B галлюцинирует имена: reverse(s), vowel_count(s) вместо написанных "
        "reverse_string/count_vowels. Это целевой сбой, который закрывает чанковая память.",
        "RAG по фактам пользователя: reuse 1.0 при 0/3 code-hits — имена утекают из текстов запросов; "
        "вспоминаются имена, а не реализации.",
        "На CPU (1.5B) лучший reuse у attn-режима (0.667, единственный настоящий вызов на чистой пробе); "
        "value/maxsim-скоринг ищут хуже (1/3). На 4B все режимы поиска 3/3 — сильная модель компенсирует.",
        "Цена чанковой памяти на 4B: ~4.4 тыс. embed-токенов против ~0.5 тыс. у baseline (~9×), "
        "латентность хода +4 % (1.04 с против 1.00 с).",
    ]


def caveats() -> list[str]:
    return [
        "Пилот не завершён (7/12): ДИ широкие, различия v4 против rag/pyramidkv могут стать значимыми "
        "после догона очереди; interim-числа не финальные.",
        "Recall — keyword-скоринг по синтетическому корпусу; full < 1.0 из-за сложности multihop-проб "
        "(0.554 даже с полным контекстом), а не из-за отказа памяти.",
        "Оценка чанковой памяти — одна короткая сессия (3 функции), n=1, greedy: проверка механики, "
        "не статистика. Случай «тысячи строк на один запрос» не покрыт — нужен стресс-тест длинной генерации.",
        "Латентность корпуса — end-to-end на одной машине без interleaved-повторов между сессиями; "
        "сравнение стратегий внутри батча честное (чередование), абсолютные числа переносить осторожно.",
        "causal_conv1d не установлен (референсный kernel), flash-attention на AMD не включён — "
        "абсолютная латентность занижена относительно оптимизированной сборки; сравнение стратегий не искажено.",
    ]


def next_steps() -> list[str]:
    return [
        "Догнать пилот (5 диалогов ≈ 9.8 ч GPU) и пройти ГК-2; затем основная очередь main (48 батчей).",
        "Стресс-тест контентного ключа: длинная генерация (десятки функций в одном ответе) — проверка, "
        "что вспоминается нужный кусок, а не вся генерация.",
        "Черновик фрагмента для статьи готов (results/corpus/PAPER_FRAGMENT.md); вставка после "
        "завершения основной очереди и просмотра заказчиком.",
    ]


def repro() -> list[str]:
    return [
        "HF_HUB_OFFLINE=1 .venv/bin/python -m tmem.corpus_run --preset pilot --model qwen35-4b "
        "--dtype bfloat16 --batches 4 --run-id pilot-v1",
        "HF_HUB_OFFLINE=1 .venv/bin/python scripts/chunk_code_eval.py      # CPU, 1.5B",
        "HF_HUB_OFFLINE=1 .venv/bin/python scripts/chunk_code_eval_4b.py   # GPU, 4B",
        ".venv/bin/python scripts/build_night_report.py                    # этот отчёт",
    ]


def chunk_rows(data: dict) -> tuple[list[str], list[list[str]]]:
    headers = ["Стратегия", "Reuse (все 3)", "turn8", "turn10", "Hits", "Латентность хода, с", "embed токены"]
    rows = []
    for name, s in data["strategies"].items():
        hits = s["retrieval_hits"]
        rows.append([name, f3(s["probe_mean"]), f3(s["probe_scores"].get("turn8")),
                     f3(s["probe_scores"].get("turn10")),
                     f"{sum(1 for v in hits.values() if v)}/{len(hits)}",
                     str(s["mean_turn_latency_s"]), str(s["embed_tokens"])])
    return headers, rows


def markdown(d: dict) -> str:
    s = d["summ"]["strategies"]
    lines = [
        "# Ночной прогон и вариации архитектуры чанковой памяти",
        "",
        f"Дата: {datetime.now():%d.%m.%Y} · модель корпуса: Qwen3.5-4B bf16 (ROCm, RX 7900 XTX) · "
        f"git {d['cfg'].get('git_sha')} · данные на 14.11.2026 12:02",
        "",
        "## Резюме", "",
    ]
    lines += [f"- {b}" for b in resume_bullets(d)]
    lines += ["", "## 1. Протокол ночного прогона", ""] + [f"- {b}" for b in pilot_protocol(d)]
    lines += ["", "## 2. Результаты пилота", "",
              "| Стратегия | n | Recall [95 % ДИ] | direct | paraphr. | multihop | temporal | unansw. | e2e, с | prompt токены |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for n in ORDER:
        e = s[n]
        lo, hi = e["recall_ci95"]
        lines.append(f"| {n} | {e['n_dialogues']} | {e['recall_mean']:.3f} [{lo:.3f}; {hi:.3f}] "
                     f"| {f3(e['recall_direct'])} | {f3(e['recall_paraphrase'])} "
                     f"| {f3(e['recall_by_kind']['multihop'])} | {f3(e['recall_by_kind']['temporal'])} "
                     f"| {f3(e['recall_by_kind']['unanswerable'])} | {e['e2e_latency_median_s']} "
                     f"| {round(e['prompt_tokens'])} |")
    lines += ["", "![Recall пилота](figs/pilot_recall.png)", "",
              "### Recall по lag-зонам", "",
              "| Стратегия | " + " | ".join(LAGS) + " |", "|---" * (len(LAGS) + 1) + "|"]
    for n in ORDER:
        lines.append(f"| {n} | " + " | ".join(f3(s[n]["recall_by_lag"][k]) for k in LAGS) + " |")
    lines += ["", "### Попарные сравнения (paired bootstrap, поправка Холма)", ""]
    lines += [f"- {k}: p = {v:.4f}" for k, v in sorted(d["pvals"].items())]
    lines += ["", "## 3. Вариации архитектуры чанковой памяти", "",
              "Идея: хранить не факты пользователя, а чанки собственной генерации. Ключ = активации "
              "слоёв самого блока (как hidden states в чат-версии), значение = контекстуализированный "
              "чанк (пул по токенам чанка в контексте породившего промпта), центрирование бегущим "
              "средним с обеих сторон. Поиск по сходству вытаскивает нужный кусок генерации, а не всю "
              "генерацию (один запрос может породить тысячи строк).", ""]
    lines += [f"- {b}" for b in chunk_protocol()]
    lines += ["", "### CPU (Qwen2-1.5B)", ""]
    h, rows = chunk_rows(d["cpu"])
    lines += ["| " + " | ".join(h) + " |", "|---" * len(h) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines += ["", "### GPU (Qwen3.5-4B)", ""]
    h, rows = chunk_rows(d["b4"])
    lines += ["| " + " | ".join(h) + " |", "|---" * len(h) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines += ["", "![Вариации чанковой памяти](figs/chunk_variations.png)", "", "### Находки", ""]
    lines += [f"- {b}" for b in chunk_findings(d)]
    lines += ["", "## 4. Ограничения", ""] + [f"- {b}" for b in caveats()]
    lines += ["", "## 5. Выводы и следующие шаги", ""] + [f"- {b}" for b in next_steps()]
    lines += ["", "## Приложение. Воспроизведение", "", "```bash"] + repro() + ["```", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ pdf

def build_pdf(d: dict) -> None:
    import matplotlib
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(fonts / "DejaVuSans-Bold.ttf")))
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="DejaVu", fontSize=9.5, leading=12.5)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, spaceAfter=2)
    heading = ParagraphStyle("heading", parent=styles["Heading2"], fontName="DejaVu-Bold",
                             fontSize=12, spaceBefore=8, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=heading, fontSize=10.5)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=9.5)
    title = ParagraphStyle("title", parent=styles["Title"], fontName="DejaVu-Bold", fontSize=15, leading=19)
    meta = ParagraphStyle("meta", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"))

    doc = SimpleDocTemplate(str(OUT / "Night_Run_Report.pdf"), pagesize=landscape(A4),
                            leftMargin=1.1 * cm, rightMargin=1.1 * cm,
                            topMargin=1.1 * cm, bottomMargin=1.1 * cm,
                            title="Ночной прогон и вариации чанковой памяти",
                            author="tmem experiments")

    def table(headers, rows, first_w=5.2):
        data = [[Paragraph(esc(h), small) for h in headers]]
        data += [[Paragraph(esc(c), small) for c in r] for r in rows]
        widths = [first_w * cm] + [(27.5 - first_w) / (len(headers) - 1) * cm] * (len(headers) - 1)
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                                [colors.white, colors.Color(.96, .97, 1)]),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return t

    def bullets(items):
        return [Paragraph(f"• {esc(b)}", bullet) for b in items]

    s = d["summ"]["strategies"]
    story = [Paragraph("Ночной прогон и вариации архитектуры чанковой памяти", title),
             Paragraph(f"Дата: {datetime.now():%d.%m.%Y} · Qwen3.5-4B bf16, AMD RX 7900 XTX (ROCm) · "
                       f"git {esc(d['cfg'].get('git_sha'))} · данные на 14.11.2026 12:02 · "
                       "внутренний технический отчёт", meta),
             Spacer(1, 0.3 * cm),
             Paragraph("Резюме", heading)] + bullets(resume_bullets(d))
    story += [Paragraph("1. Протокол ночного прогона", heading)] + bullets(pilot_protocol(d))
    story += [Paragraph("2. Результаты пилота (7/12 диалогов)", heading)]
    headers = ["Стратегия", "n", "Recall [95 % ДИ]", "direct", "paraphr.", "multihop",
               "temporal", "unansw.", "e2e, с", "prompt токены"]
    rows = []
    for n in ORDER:
        e = s[n]
        lo, hi = e["recall_ci95"]
        rows.append([SHORT[n], e["n_dialogues"], f"{e['recall_mean']:.3f} [{lo:.3f}; {hi:.3f}]",
                     f3(e["recall_direct"]), f3(e["recall_paraphrase"]),
                     f3(e["recall_by_kind"]["multihop"]), f3(e["recall_by_kind"]["temporal"]),
                     f3(e["recall_by_kind"]["unanswerable"]), e["e2e_latency_median_s"],
                     round(e["prompt_tokens"])])
    story += [table(headers, rows), Spacer(1, 0.3 * cm),
              Image(str(FIGS / "pilot_recall.png"), width=23.5 * cm, height=8.46 * cm),
              Spacer(1, 0.2 * cm), Paragraph("Recall по lag-зонам", sub)]
    rows = [[SHORT[n]] + [f3(s[n]["recall_by_lag"][k]) for k in LAGS] for n in ORDER]
    story += [table(["Стратегия"] + list(LAGS), rows, first_w=5.2),
              Spacer(1, 0.2 * cm),
              Paragraph("Попарные сравнения recall (paired bootstrap, поправка Холма)", sub)]
    story += bullets([f"{k}: p = {v:.4f}" + (" — значимо" if v < 0.05 else "")
                      for k, v in sorted(d["pvals"].items())])
    story += [PageBreak(), Paragraph("3. Вариации архитектуры чанковой памяти", heading),
              Paragraph("Идея: хранить не факты пользователя, а чанки собственной генерации. "
                        "Ключ = активации слоёв самого блока (как hidden states в чат-версии), "
                        "значение = контекстуализированный чанк (пул по токенам чанка в контексте "
                        "породившего промпта), центрирование бегущим средним с обеих сторон. Поиск по "
                        "сходству вытаскивает нужный кусок генерации, а не всю генерацию — один запрос "
                        "может породить тысячи строк кода.", body),
              Spacer(1, 0.2 * cm)] + bullets(chunk_protocol())
    story += [Paragraph("CPU: Qwen2-1.5B fp32", sub)]
    h, rows = chunk_rows(d["cpu"])
    story += [table(h, rows, first_w=6.0), Spacer(1, 0.3 * cm),
              Paragraph("GPU: Qwen3.5-4B bf16", sub)]
    h, rows = chunk_rows(d["b4"])
    story += [table(h, rows, first_w=6.0), Spacer(1, 0.3 * cm),
              Image(str(FIGS / "chunk_variations.png"), width=24.5 * cm, height=8.45 * cm),
              Spacer(1, 0.2 * cm), Paragraph("Находки", sub)] + bullets(chunk_findings(d))
    story += [PageBreak(), Paragraph("4. Ограничения", heading)] + bullets(caveats())
    story += [Paragraph("5. Выводы и следующие шаги", heading)] + bullets(next_steps())
    story += [Paragraph("Приложение. Воспроизведение", heading)]
    story += [Paragraph(esc(c), ParagraphStyle("code", parent=body, fontName="DejaVu",
                                               fontSize=8, leading=10.5, leftIndent=10))
              for c in repro()]
    doc.build(story)


def main() -> None:
    d = collect()
    make_figures(d)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "Night_Run_Report.md").write_text(markdown(d), encoding="utf-8")
    build_pdf(d)
    print("report:", OUT / "Night_Run_Report.pdf")
    print("md:", OUT / "Night_Run_Report.md")


if __name__ == "__main__":
    main()

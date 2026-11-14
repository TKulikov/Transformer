"""Ночной вотчер корпусных прогонов (заказ от 13.11.2026: отчёт к 07:00).

Что делает:
1. Дожидается завершения текущего раннера (--batches 4, ГК-1) и проверяет санити.
2. Продолжает очередь pilot-v1 батчами до 06:30 (resume по манифесту, атомарно).
3. Если весь пилот (12 диалогов) завершён без ошибок и воспроизводит порядки
   recall из статьи — до 04:30 запускает основную очередь main-v1 (шаг 4 плана).
4. После каждого вызова раннера обновляет results/corpus/MORNING_REPORT.md
   (краткий отчёт) — он существует даже при аварийной остановке вотчера.
5. К 06:50 GPU освобождается (SIGINT текущему батчу — он перезапустится при
   следующем resume), финальный отчёт + черновик фрагмента для статьи
   (PAPER_FRAGMENT.md) готовы до 07:00.

Запуск: setsid nohup .venv/bin/python scripts/night_watch.py \
            > results/corpus/night_watch_nohup.log 2>&1 &
Лог: results/corpus/night_watch.log; статус: results/corpus/NIGHT_STATUS.json.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tmem.corpus_stats import load_run, summarize, compare_pairs  # noqa: E402

CORPUS = ROOT / "results" / "corpus"
LOG = CORPUS / "night_watch.log"
REPORT = CORPUS / "MORNING_REPORT.md"
FRAGMENT = CORPUS / "PAPER_FRAGMENT.md"
STATUS = CORPUS / "NIGHT_STATUS.json"
PY = str(ROOT / ".venv" / "bin" / "python")

now = datetime.now()


def _next_occurrence(hhmm: str) -> datetime:
    """Ближайшее наступление времени HH:MM (сегодня, если ещё не прошло)."""
    t = datetime.strptime(hhmm, "%H:%M").time()
    candidate = datetime.combine(now.date(), t)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


# БАГ-ФИКС 14.11: наивный now+1day после полуночи сдвигал дедлайн на сутки.
RUN_STOP = _next_occurrence("06:30")
HARD_KILL = _next_occurrence("06:50")
MAIN_LATEST_START = _next_occurrence("04:30")
INITIAL_PID = 179928  # текущий прогон --batches 4

FULL = "classic_kv_full"
V4 = "v4_rag_kv"
RAG = "rag_text_only"
SLIDING = "sparse_sliding_128"
SNAPKV = "snapkv_128"
PYRAMID = "pyramidkv_128"


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_dir(run_id: str) -> Path:
    return CORPUS / run_id


def run_state(run_id: str) -> dict | None:
    rd = run_dir(run_id)
    if not (rd / "run_config.json").exists():
        return None
    run = load_run(rd)
    cfg = run["config"]
    done_ids = {r["batch_id"] for r in run["records"]}
    pending = [s["spec_id"] for s in cfg["specs"] if s["spec_id"] not in done_ids]
    walls = [r.get("wall_s") or 0 for r in run["records"]]
    manifest = {}
    mp = rd / "manifest.jsonl"
    if mp.exists():
        for line in mp.read_text().splitlines():
            rec = json.loads(line)
            manifest[rec["batch_id"]] = rec  # последняя запись побеждает
    # load_run отдаёт только done-батчи; падения ищем в манифесте:
    # failed с reruns>=1 = батч упал минимум дважды подряд
    failed = [r for r in manifest.values()
              if r.get("status") == "failed" and r.get("reruns", 0) >= 1]
    return {"run": run, "summary": summarize(run), "pending": pending,
            "avg_wall_s": (sum(walls) / len(walls)) if walls else 2700.0,
            "failed_persistent": failed, "walls": walls}


def sanity_gate(st: dict) -> list[str]:
    """Авто-ГК-1: грубые проверки до продолжения очереди."""
    problems = []
    s = st["summary"]["strategies"]
    if st["summary"]["n_dialogues_done"] == 0:
        problems.append("нет ни одного завершённого батча")
        return problems
    if st["failed_persistent"]:
        problems.append("батчи с повторными падениями: "
                        + ", ".join(r["batch_id"] for r in st["failed_persistent"]))
    if FULL in s and (s[FULL]["recall_mean"] is None or s[FULL]["recall_mean"] < 0.8):
        problems.append(f"recall полного контекста подозрительно низок: {s[FULL]['recall_mean']}")
    if V4 in s and SLIDING in s and s[V4]["recall_mean"] < s[SLIDING]["recall_mean"] - 0.05:
        problems.append(f"v4 хуже sliding window: {s[V4]['recall_mean']} < {s[SLIDING]['recall_mean']}")
    for name, entry in s.items():
        lat = entry.get("e2e_latency_median_s")
        if lat is not None and (lat != lat or lat <= 0):  # NaN / неположительная
            problems.append(f"аномальная латентность {name}: {lat}")
    return problems


def pilot_criteria(st: dict) -> tuple[bool, list[str]]:
    """Критерий успеха пилота из плана: порядки recall как в статье."""
    notes = []
    s = st["summary"]["strategies"]
    ok = True
    if st["pending"] or st["failed_persistent"]:
        return False, [f"пилот не завершён: осталось {len(st['pending'])}, "
                       f"падений {len(st['failed_persistent'])}"]
    need = [FULL, V4, RAG, SLIDING, SNAPKV, PYRAMID]
    if any(n not in s for n in need):
        return False, ["не все стратегии присутствуют в результатах"]

    def r(n):
        return s[n]["recall_mean"]

    checks = [
        (r(FULL) >= 0.9, f"full ≈ 1.0: {r(FULL):.3f}"),
        (r(V4) >= r(RAG) - 0.02, f"v4 ≥ rag_text: {r(V4):.3f} vs {r(RAG):.3f}"),
        (r(RAG) >= r(SLIDING) - 0.02, f"rag_text ≥ sliding: {r(RAG):.3f} vs {r(SLIDING):.3f}"),
        (r(V4) >= r(SNAPKV) - 0.02, f"v4 ≥ snapkv: {r(V4):.3f} vs {r(SNAPKV):.3f}"),
        (r(V4) >= r(PYRAMID) - 0.02, f"v4 ≥ pyramidkv: {r(V4):.3f} vs {r(PYRAMID):.3f}"),
    ]
    for passed, label in checks:
        notes.append(("✓ " if passed else "✗ ") + label)
        ok = ok and passed
    return ok, notes


def invoke_runner(preset: str, run_id: str, n_batches: int) -> str:
    """Один вызов раннера с таймбоксом; возвращает исход (ok/timeout/error)."""
    cmd = [PY, "-m", "tmem.corpus_run", "--preset", preset, "--model", "qwen35-4b",
           "--dtype", "bfloat16", "--batches", str(n_batches), "--run-id", run_id]
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    log(f"запуск: {' '.join(cmd)}")
    runner_log = (CORPUS / f"runner_{run_id}.log").open("a", encoding="utf-8")
    runner_log.write(f"\n=== {datetime.now():%F %T} batches={n_batches} ===\n")
    runner_log.flush()
    proc = subprocess.Popen(cmd, env=env, stdout=runner_log, stderr=subprocess.STDOUT)
    outcome = "ok"
    while True:
        try:
            proc.wait(timeout=30)
            break
        except subprocess.TimeoutExpired:
            if datetime.now() >= HARD_KILL:
                log("HARD_KILL: SIGINT раннеру (батч перезапустится при resume)")
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    proc.kill()
                outcome = "timeout"
                break
    if proc.returncode not in (0, None) and outcome == "ok":
        outcome = f"error(rc={proc.returncode})"
    runner_log.close()
    log(f"раннер завершился: {outcome}")
    return outcome


def fmt_ci(entry: dict) -> str:
    lo, hi = entry.get("recall_ci95", (None, None))
    if lo is None:
        return "—"
    return f"{entry['recall_mean']:.3f} [{lo:.3f}; {hi:.3f}]"


def _chunk_table(path: Path, title: str) -> list[str]:
    if not path.exists():
        return [f"## {title}", "",
                "- Ещё не завершено; лог: results/chunk_code_nohup.log / results/chunk_code_4b_nohup.log", ""]
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        f"## {title}",
        "",
        f"Модель: {data.get('model')} ({data.get('device')}, {data.get('dtype')}, greedy). "
        "Сценарий: 3 функции → filler → 3 пробы на переиспользование кода; окно = 2 пары "
        "(ранний код выпадает из контекста, вернуть может только память).",
        "",
        "| Стратегия | Reuse (probe mean) | Retrieval hits (из 3) | Латентность хода, с | embed токены |",
        "|---|---|---|---|---|",
    ]
    for name, s in data.get("strategies", {}).items():
        hits = s.get("retrieval_hits", {})
        n_hits = sum(1 for v in hits.values() if v)
        pm = s.get("probe_mean")
        lines.append(f"| {name} | {pm if pm is not None else '—'} | {n_hits}/{len(hits) or 3} "
                     f"| {s.get('mean_turn_latency_s', '—')} | {s.get('embed_tokens', '—')} |")
    lines += [
        "",
        "Reuse — имя нужной функции в ответе (turn9 загрязнён echo, чистые: turn8/10); "
        "retrieval hit — банк положил правильный код в промпт.",
        f"Подробности (ответы, injected notes): `{path.parent}`.",
        "",
    ]
    return lines


def chunk_code_section() -> list[str]:
    """Секции утреннего отчёта по оценке чанковой памяти (CPU и 4B)."""
    base = ROOT / "results" / "chunk_code"
    return (_chunk_table(base / "code-v1" / "summary.json",
                         "Чанковая память: задача написания кода (CPU, ключ=запрос)")
            + _chunk_table(base / "code-4b-v1" / "summary.json",
                           "Чанковая память: контентный ключ (4B, активации блока)"))


def render_report(final: bool, stop_reason: str, decisions: list[str]) -> None:
    lines = [
        "# Ночной отчёт — корпусные прогоны (к 07:00)",
        "",
        f"Обновлён: {datetime.now():%Y-%m-%d %H:%M} · статус: "
        + ("**ФИНАЛЬНЫЙ**" if final else "промежуточный (вотчер работает)"),
        "",
        f"Причина остановки/режим: {stop_reason}",
        "",
    ]
    for d in decisions:
        lines.append(f"- {d}")
    lines.append("")
    for run_id in ("pilot-v1", "main-v1"):
        st = run_state(run_id)
        if st is None:
            continue
        summ = st["summary"]
        done, total = summ["n_dialogues_done"], summ["n_dialogues_total"]
        gpu_h = sum(st["walls"]) / 3600
        lines += [
            f"## {run_id} ({summ.get('preset')})",
            "",
            f"- Прогресс: **{done}/{total}** диалогов · GPU-время: {gpu_h:.2f} ч · "
            f"средний батч: {st['avg_wall_s'] / 60:.1f} мин",
            f"- Осталось в очереди: {len(st['pending'])}"
            + (f" (≈ {len(st['pending']) * st['avg_wall_s'] / 3600:.1f} ч)" if st["pending"] else ""),
            f"- Повторные падения: {len(st['failed_persistent'])}",
            "",
            "| Стратегия | Recall [95 % ДИ] | e2e медиана, с | prompt токены |",
            "|---|---|---|---|",
        ]
        for name, s in summ["strategies"].items():
            lat = s["e2e_latency_median_s"]
            pt = s["prompt_tokens"]
            lines.append(f"| {name} | {fmt_ci(s)} | {lat if lat is not None else '—'} "
                         f"| {round(pt) if pt is not None else '—'} |")
        lines.append("")
        if done >= 5:
            pvals = compare_pairs(st["run"], [
                (V4, RAG), (V4, PYRAMID), (V4, FULL), (RAG, SLIDING)])
            if pvals:
                lines += ["Попарные сравнения recall (paired bootstrap, поправка Холма):", ""]
                lines += [f"- {k}: p = {'**' if p < 0.05 else ''}{p:.4f}{'**' if p < 0.05 else ''}"
                          for k, p in sorted(pvals.items())]
                lines.append("")
        if V4 in summ["strategies"]:
            lag = summ["strategies"][V4]["recall_by_lag"]
            lines += ["v4_rag_kv по lag-зонам: "
                      + ", ".join(f"{k}: {v:.3f}" if v is not None else f"{k}: —"
                                  for k, v in lag.items()), ""]
        lines += [f"Подробности: `results/corpus/{run_id}/INTERIM.md`", ""]
    lines += chunk_code_section()
    lines += [
        "## Что дальше",
        "",
        "- Решение по ГК-1/ГК-2 — за заказчиком (см. план docs/PLAN-fullscale-tests.md).",
        "- Черновик фрагмента для статьи: `results/corpus/PAPER_FRAGMENT.md`.",
        "- Продолжить прогоны: та же команда corpus_run (resume по манифесту).",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    # --- черновик фрагмента для статьи (только числа из того же стат-ядра) ---
    st = run_state("main-v1") or run_state("pilot-v1")
    if st and st["summary"]["strategies"]:
        summ = st["summary"]
        src = "main-v1" if (CORPUS / "main-v1" / "run_config.json").exists() else "pilot-v1"
        frag = [
            "# ФРАГМЕНТ ДЛЯ СТАТЬИ (черновик, требует просмотра заказчиком)",
            "",
            f"Источник чисел: `results/corpus/{src}` (то же стат-ядро, что INTERIM.md).",
            "Вставка: новый подраздел раздела результатов; править scripts/paper_text.py,",
            "затем `.venv/bin/python scripts/build_paper.py` и `scripts/verify_paper.py`.",
            "Статус данных: ПРЕДВАРИТЕЛЬНЫЕ (пилот/часть основной очереди), не финальная оценка.",
            "",
            "## Предварительная многодиалоговая оценка",
            "",
            f"Протокол: корпус синтетических диалогов ({summ['n_dialogues_done']} завершённых диалогов,"
            " 100 ходов, greedy, max_new_tokens=32), 7 стратегий, бутстреп-ДИ по диалогам,"
            " латентность end-to-end по 5 чередующимся повторам после warmup.",
            "",
            "| Система | Recall [95 % ДИ] | Латентность медиана, с |",
            "|---|---|---|",
        ]
        for name, s in summ["strategies"].items():
            lat = s["e2e_latency_median_s"]
            frag.append(f"| {name} | {fmt_ci(s)} | {lat if lat is not None else '—'} |")
        frag += [
            "",
            "Ограничения: синтетический корпус, одна модель (Qwen3.5-4B bf16, ROCm),",
            "неполная очередь; числа могут измениться после завершения основного прогона.",
            "",
        ]
        FRAGMENT.write_text("\n".join(frag), encoding="utf-8")

    STATUS.write_text(json.dumps({
        "updated": datetime.now().isoformat(timespec="seconds"),
        "final": final, "stop_reason": stop_reason, "decisions": decisions,
        "runs": {rid: {"done": (run_state(rid) or {}).get("summary", {}).get("n_dialogues_done"),
                       "pending": len((run_state(rid) or {}).get("pending", []))}
                 for rid in ("pilot-v1", "main-v1") if run_state(rid)},
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def wait_initial() -> None:
    log(f"жду завершения исходного раннера pid={INITIAL_PID}")
    last_render = datetime.now()
    while True:
        alive = False
        try:
            os.kill(INITIAL_PID, 0)
            with open(f"/proc/{INITIAL_PID}/cmdline", "rb") as fh:
                alive = b"corpus_run" in fh.read()
        except (ProcessLookupError, FileNotFoundError):
            alive = False
        if not alive:
            break
        if datetime.now() >= HARD_KILL:
            log("HARD_KILL: исходный раннер не успел до 06:50 — SIGINT (незавершённый батч перезапустится при resume)")
            try:
                os.kill(INITIAL_PID, signal.SIGINT)
            except ProcessLookupError:
                pass
            time.sleep(120)
            break
        if (datetime.now() - last_render).total_seconds() > 1800:
            render_report(False, "исходный раннер ещё работает (первые 4 батча)", [])
            last_render = datetime.now()
        time.sleep(60)
    log("исходный раннер завершён")


def main() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    decisions: list[str] = []
    stop_reason = "очередь завершена"
    wait_initial()

    st = run_state("pilot-v1")
    if st is None:
        stop_reason = "pilot-v1 не найден — нечего продолжать"
        render_report(True, stop_reason, decisions)
        return
    problems = sanity_gate(st)
    if problems:
        stop_reason = "авто-ГК-1 НЕ пройден: " + "; ".join(problems)
        decisions.append("ОЧЕРЕДЬ ОСТАНОВЛЕНА на sanity-чеке — требуется разбор причин.")
        log(stop_reason)
        render_report(True, stop_reason, decisions)
        return
    decisions.append(f"Авто-ГК-1 пройден ({st['summary']['n_dialogues_done']} батчей): "
                     "recall/латентность в ожидаемых диапазонах.")
    log("санити ОК, продолжаю очередь pilot-v1")

    active = "pilot-v1"
    while datetime.now() < RUN_STOP:
        st = run_state(active)
        if st is None:
            break
        if st["failed_persistent"]:
            stop_reason = "повторные падения батчей: " + ", ".join(
                r["batch_id"] for r in st["failed_persistent"])
            decisions.append("ОЧЕРЕДЬ ОСТАНОВЛЕНА: батч падает повторно — нужен разбор.")
            break
        if not st["pending"]:
            if active == "pilot-v1":
                passed, notes = pilot_criteria(st)
                decisions.append("Критерий пилота: " + ("ВЫПОЛНЕН" if passed else "НЕ выполнен")
                                 + " (" + "; ".join(notes) + ")")
                log(f"критерий пилота: {passed}")
                if passed and datetime.now() < MAIN_LATEST_START:
                    active = "main-v1"
                    decisions.append("Пилот успешен — запущена основная очередь main-v1 (шаг 4 плана).")
                    log("переход к main-v1")
                    continue
                stop_reason = ("пилот завершён; основная очередь не запущена: "
                               + ("критерий не выполнен" if not passed else "слишком поздно (после 04:30)"))
                break
            stop_reason = "основная очередь завершена"
            break
        remaining = (RUN_STOP - datetime.now()).total_seconds()
        if remaining < 0.6 * st["avg_wall_s"]:
            stop_reason = f"до 06:30 осталось {remaining / 60:.0f} мин — меньше батча, стоп"
            break
        n = max(1, min(3, int(remaining / st["avg_wall_s"]) - 1, len(st["pending"])))
        outcome = invoke_runner(active.split("-")[0], active, n)
        render_report(False, f"в работе ({active})", decisions)
        if outcome == "timeout":
            stop_reason = "таймбокс 06:50 — GPU освобождён, незавершённый батч перезапустится при resume"
            break
        if outcome.startswith("error"):
            st2 = run_state(active)
            if st2 and st2["failed_persistent"]:
                stop_reason = "раннер упал, есть повторно падающие батчи — стоп"
                break
            decisions.append(f"вызов раннера завершился с {outcome}; пробую продолжить (resume).")

    render_report(True, stop_reason, decisions)
    log(f"ФИНАЛ: {stop_reason}; отчёт: {REPORT}")


if __name__ == "__main__":
    main()

"""Goal-directedness тест: живая память удерживает ЦЕЛЬ диалога (заказ 15.11).

Гипотеза автора: слияние близких векторов актуальных мыслей (EMA merge)
поддерживает главную мысль и направление («куда идём, что преследуем»).
Ключевое отличие от thread-теста: там главная нить была сказана ОДИН раз и
затухала; здесь ЦЕЛЬ РЕКУРРЕНТНА (повторяется) → merge подкрепляет её слот →
живая память должна удерживать цель лучше, чем без памяти или статическая.

Цель: запуск MVP к пятнице (рекуррентная). Тангенты: Rust-перепись, маркетинг,
логотип. Пробы спрашивают цель/фокус; expected=ключи цели, stale=ключи тангента.
Метрики: goal recall (цель в ответе), goal confusion (тангент вместо цели), tone.

Условия (центрированный RAG last, окно 2):
* v3_clean      — без живой памяти (базлайн);
* living_merge  — memory=True, merge_k=0.25, TTL on (подкрепление цели);
* living_nomerge— memory=True, merge_k=None (v1 mean), TTL on;
* static_merge  — memory=True, merge_k=0.25, TTL off (статический банк).
Изолирует эффект слияния (merge vs nomerge) и TTL (living vs static).

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/goal_dialogue_test.py
Выход:  results/goal_dialogue/{summary.json,REPORT.md,<cond>.json}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from tmem.dialogue import Strategy, run_session  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.qa import DialogSession, Probe  # noqa: E402
from tmem.stress import analyze_session  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "goal_dialogue"
GOAL = ["mvp", "пятниц", "запуск"]


def build_goal_dialogue() -> DialogSession:
    turns: list[str] = []
    probes: list[Probe] = []

    def say(t):
        turns.append(t)

    def probe(q, expected, stale, plant=0):
        idx = len(turns)
        turns.append(q)
        probes.append(Probe(after_turn=idx, question=q, expected=list(expected),
                            stale=list(stale), plant_turn=plant))

    # --- завязка: цель ---
    say("Привет! Помоги подготовить запуск нашего MVP к пятнице, это главное сейчас.")  # 0
    say("Нам нужно успеть выкатить минимальную версию продукта именно к пятнице.")  # 1
    say("Давай держать фокус на запуске MVP, времени мало.")  # 2

    # --- тангент A: Rust ---
    say("Кстати, я тут подумал, может переписать бэкенд на Rust, говорят он быстрый.")  # 3
    say("Коллега хвалит Rust, borrow checker правда сложный, но производительность.")  # 4
    say("Я даже начал смотреть туториалы по Rust, интересно.")  # 5
    probe("Стоп, а какова наша ГЛАВНАЯ цель на этой неделе, напомни?",
          GOAL, ["rust", "перепис"], plant=0)  # 6
    say("Точно, запуск MVP к пятнице, а не Rust. Rust потом, сейчас не до него.")  # 7

    # --- рекурренция цели + тангент B: маркетинг ---
    say("Вернёмся к запуску MVP. Что у нас по готовности продукта к пятнице?")  # 8
    say("Ой, а давай ещё подумаем про маркетинг, рекламу запустим, таргетинг.")  # 9
    say("Можно нанять маркетолога, сделать воронку, посчитать бюджет на рекламу.")  # 10
    say("Маркетинг это важно, я даже презентацию набросал для инвесторов про рекламу.")  # 11
    probe("Так, а на чём мы должны ФОКУСИРОВАТЬСЯ прямо сейчас, что главное?",
          GOAL, ["маркетинг", "рекла", "таргет"], plant=0)  # 12
    say("Да, фокус на запуске MVP к пятнице, маркетинг после запуска.")  # 13

    # --- рекурренция цели + тангент C: логотип ---
    say("Итак, запуск MVP к пятнице в приоритете. Что осталось доделать?")  # 14
    say("А ещё я хочу новый логотип сделать, дизайн бренда, цвета подобрать.")  # 15
    say("Логотип должен быть минималистичный, я референсы собрал, шрифты смотрю.")  # 16
    say("Дизайнер предложил три варианта логотипа, выбираю.")  # 17
    probe("Напомни, что мы преследуем, к какому сроку и что именно запускаем?",
          GOAL, ["логотип", "дизайн", "бренд"], plant=0)  # 18
    say("Запускаем MVP к пятнице, логотип подождёт, сейчас не главное.")  # 19

    # --- рекурренция + сборка ---
    say("Хорошо, держим курс на запуск MVP к пятнице. Давай список того что успеть.")  # 20
    say("Главное чтобы к пятнице MVP работал, остальное потом.")  # 21
    say("Я немного устал, неделя тяжёлая, но запуск MVP важен.")  # 22
    probe("Собери в кучу: что мы запускаем, к какому сроку, и что сейчас в приоритете?",
          GOAL, ["rust", "маркетинг", "логотип"], plant=0)  # 23
    say("Да, запуск MVP к пятнице, фокус на продукте, не на Rust/маркетинге/логотипе.")  # 24

    # --- финальный дрейф + возврат ---
    say("Кстати про логотип, я всё же выбрал синий вариант, но это не срочно.")  # 25
    say("И маркетинг потом, после того как выкатим продукт.")  # 26
    say("Ладно, вернёмся к делу, что там по готовности MVP?")  # 27
    probe("И последнее: наша главная цель и срок, чтобы я не забыл куда иду?",
          GOAL, ["логотип", "маркетинг", "rust"], plant=0)  # 28
    say("Запуск MVP к пятнице. Спасибо, что помог не потерять фокус.")  # 29
    say("Пойду доделывать продукт, чтобы успеть к пятнице.")  # 30

    return DialogSession(name=f"goal-{len(turns)}t-{len(probes)}q",
                         opening=turns, fillers=[], probes=probes)


CONDITIONS = [
    ("v3_clean", dict(memory=False, memory_kv=False)),
    ("living_merge", dict(memory=True, memory_merge_k=0.25, memory_decay=True,
                          memory_crit=0.95, memory_window_crit=0.5)),
    ("living_nomerge", dict(memory=True, memory_merge_k=None, memory_decay=True,
                            memory_crit=0.95, memory_window_crit=0.5)),
    ("static_merge", dict(memory=True, memory_merge_k=0.25, memory_decay=False,
                          memory_crit=0.95, memory_window_crit=0.5)),
]


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    session = build_goal_dialogue()
    print(f"диалог: {len(session.user_turns())} ходов, {len(session.probes)} проб", flush=True)
    summary = {"model": "qwen35-4b", "device": device, "window": 2,
               "turns": len(session.user_turns()), "n_probes": len(session.probes),
               "conditions": {}}
    for name, mem in CONDITIONS:
        strat = Strategy(name, context="window", window=2, rag=True, rag_lambda=0.0,
                         rag_centered=True, rag_layer="last", **mem)
        t0 = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        run = run_session(bundle, session, strat, max_new_tokens=48, seed=0)
        a = analyze_session({name: run})[name]
        entry = {"goal_recall": a["recall_mean"], "goal_confusion": a["stale_rate"],
                 "tone": a.get("tone_mean"), "wall_s": round(time.perf_counter() - t0, 1),
                 "probes": [{"q": d["question"], "score": d["score"], "stale": d["stale_score"],
                             "answer": d["answer"]} for d in a["probe_details"]]}
        summary["conditions"][name] = {k: v for k, v in entry.items() if k != "probes"}
        (DIR / f"{name}.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  goal_recall={a['recall_mean']} goal_confusion={a['stale_rate']} tone={a.get('tone_mean')}", flush=True)
    write_report(summary)
    print("готово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Goal-directedness: живая память удерживает ЦЕЛЬ (рекуррентную)",
        "",
        f"Диалог: {s['turns']} ходов, {s['n_probes']} проб. Цель: запуск MVP к пятнице (рекуррентна).",
        "Тангенты: Rust / маркетинг / логотип. goal_recall — цель в ответе; goal_confusion — тангент вместо цели.",
        "Гипотеза: merge подкрепляет рекуррентную цель → living_merge лучше удерживает направление.",
        "",
        "| Условие | goal_recall | goal_confusion | tone |",
        "|---|---|---|---|",
    ]
    for name, c in s["conditions"].items():
        tone = f"{c['tone']:.3f}" if c.get("tone") is not None else "—"
        lines.append(f"| {name} | {c['goal_recall']:.3f} | {c['goal_confusion']:.3f} | {tone} |")
    lines += ["", "v3_clean — без живой памяти; living_merge — merge_k=0.25+TTL; living_nomerge — v1 mean+TTL;",
              "static_merge — merge без TTL. Ниже goal_confusion = лучше удержание цели. Данные: `<cond>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Thread-coherence тест: реалистичный меандрирующий диалог с дрейфом темы.

Заказ 15.11: проверить назначение латентного канала — удержание нити диалога
при коротком окне и фактах от RAG. battery100 (факт-пробы) этого не стрессует:
модель и так когерентна. Здесь — человеческий диалог с ответвлениями по смыслу,
прыжками логики и возвратами.

Главная нить: поездка в Хельсинки на конференцию 14 марта, отель HG-77120,
кот Барсик. Ответвления вводят КОНФУЗЯЩИЕ значения (Берлин/Адлон, дедлайн
20 марта, кот друга Мурзик). Пробы после каждого ответвления спрашивают
значение ГЛАВНОЙ нити; ошибка = модель отвечает значением из недавнего
тангента (потеря нити). Метрики: recall (верное значение нити), confusion
(stale — значение тангента в ответе), tone.

Сравниваются формы латентного канала при одинаковом центрированном RAG на
mid-слое и коротком окне (window=2): none / additive (memory=True, v1) /
K/V (memory_kv=True, v4).

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/thread_coherence_test.py
Выход:  results/thread_coherence/{summary.json,REPORT.md,<cond>.json}
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
DIR = ROOT / "results" / "thread_coherence"


def build_thread_dialogue() -> DialogSession:
    turns: list[str] = []
    probes: list[Probe] = []

    def say(text):
        turns.append(text)

    def probe(question, expected, stale=None, plant_turn=None):
        idx = len(turns)
        turns.append(question)
        probes.append(Probe(after_turn=idx, question=question, expected=list(expected),
                            stale=list(stale or []), plant_turn=plant_turn))

    # --- главная нить (завязка) ---
    say("Привет! Помоги собраться в поездку — я еду в Хельсинки на конференцию 14 марта.")  # 0
    say("Отель уже забронировал, номер брони HG-77120, запиши, чтобы не потерялся.")  # 1
    say("Дома кот Барсик, надо решить, с кем его оставить на неделю.")  # 2
    say("Говорят, там в марте ещё холодно, думаю, что из одежды брать.")  # 3
    say("В общем, помоги ничего не забыть, я вечно что-то упускаю.")  # 4

    # --- ответвление 1: история про Берлин (конфузит город/отель) ---
    say("Ой, кстати, я ж в прошлом году в Берлин ездил, тоже на конференцию. Там минус пять было, я не ожидал.")  # 5
    say("Жил в Адлоне, прямо у Бранденбургских ворот, вид шикарный.")  # 6
    say("Друг Клаус меня таскал по музеям, я до сих пор под впечатлением.")  # 7
    say("А потом снег пошёл, я все ботинки промочил, жесть.")  # 8
    say("Там ещё кофе ихний крепкий, я потом три дня не спал нормально.")  # 9
    probe("Слушай, а в какой город Я сейчас собираюсь-то на конференцию?",
          ["хельсинки", "helsinki"], ["берлин", "berlin"], plant_turn=0)  # 10
    probe("А номер брони моего отеля напомни, а то вылетело из головы.",
          ["77120"], ["адлон", "adlon"], plant_turn=1)  # 11
    say("Да, точно, Хельсинки. Ладно, вернёмся к сборам.")  # 12

    # --- ответвление 2: работа (прыжок логики, конфузит дату) ---
    say("Меня тут на работе отвлекли — парсер на Rust не собирается, Oxidum версии 0.4.2, дедлайн 20 марта.")  # 13
    say("Borrow checker ругается, час уже бьюсь, коллега советует переписать на Arc.")  # 14
    say("Программирование иногда выматывает, если честно.")  # 15
    say("Ладно, с работой потом разберусь, не сейчас.")  # 16
    probe("Так, а какого числа МОЯ конференция в Хельсинки, напомни?",
          ["14"], ["20"], plant_turn=0)  # 17
    say("Точно, 14-го. А то я уже запутался в этих датах.")  # 18

    # --- ответвление 3: рецепт + кот друга (конфузит имя кота) ---
    say("Кстати, я тут фалафель научился делать — нут замачивать 8 часов, потом молоть крупно.")  # 19
    say("У друга кот Мурзик вечно лезет на стол, смешной такой, рыжий.")  # 20
    say("Я ему рецепт скинул, он тоже хочет попробовать приготовить.")  # 21
    say("Еда вообще тема, я бы ещё борщ сварил на выходные.")  # 22
    probe("А как МОЕГО кота зовут, которого я дома оставляю?",
          ["барсик"], ["мурзик"], plant_turn=2)  # 23
    say("Барсик, да. Надо же с кем-то его оставить, а то он один заскучает.")  # 24

    # --- ответвление 4: эмоциональное отступление + дрейф ---
    say("Что-то я устал, неделя тяжёлая была, голова кругом.")  # 25
    say("Иногда хочется всё бросить и уехать куда-нибудь подальше.")  # 26
    say("Но ты прав, надо доделать дела и спокойно собраться.")  # 27
    probe("Давай соберём всё в кучу: куда я еду, зачем, и что мне нужно решить дома с котом?",
          ["хельсинки", "конференц", "барсик"], ["берлин", "мурзик"], plant_turn=0)  # 28

    # --- возврат к сборам + ещё дрейф ---
    say("Так, что там по одежде — куртку тёплую брать обязательно?")  # 29
    say("Я шарф потерял в прошлом году, надо новый купить.")  # 30
    say("Кстати, в Хельсинки что посмотреть, если будет свободный вечер?")  # 31
    say("Я музеи люблю, но после Берлина уже насмотрелся, хочется чего-то спокойного.")  # 32
    say("Может, просто погуляю у воды, я такое люблю.")  # 33
    probe("Напомни номер брони отеля в Хельсинки, я хочу перепроверить.",
          ["77120"], ["адлон", "adlon"], plant_turn=1)  # 34
    probe("И что мне нужно сделать с котом перед отъездом?",
          ["барсик", "кот", "оставить"], ["мурзик"], plant_turn=2)  # 35
    say("Да, надо договориться с кем-то из друзей, кто зайдёт кормить.")  # 36
    say("Ладно, спасибо, что помог разложить всё по полочкам, а то я запутался.")  # 37
    say("Пойду дальше собираться, ещё куча дел.")  # 38

    return DialogSession(name=f"thread-{len(turns)}t-{len(probes)}q",
                         opening=turns, fillers=[], probes=probes)


CONDITIONS = [
    ("v3_clean", dict(rag_layer="mid", memory=False, memory_kv=False)),
    ("v3_clean_last", dict(rag_layer="last", memory=False, memory_kv=False)),
    ("additive_v1", dict(rag_layer="mid", memory=True, memory_window_crit=0.5,
                         memory_crit=0.95, memory_merge_k=0.25)),
    ("kv_v4", dict(rag_layer="mid", memory_kv=True, memory_window_crit=0.5,
                   memory_crit=0.95, memory_merge_k=0.25, kv_slots=32)),
]


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    session = build_thread_dialogue()
    print(f"диалог: {len(session.user_turns())} ходов, {len(session.probes)} проб", flush=True)

    summary = {"model": "qwen35-4b", "device": device,
               "window": 2, "turns": len(session.user_turns()),
               "n_probes": len(session.probes), "conditions": {}}
    for name, cfg in CONDITIONS:
        strat = Strategy(name, context="window", window=2, rag=True, rag_lambda=0.0,
                         rag_centered=True, **cfg)
        t0 = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        run = run_session(bundle, session, strat, max_new_tokens=48, seed=0)
        a = analyze_session({name: run})[name]
        entry = {"recall": a["recall_mean"], "confusion_stale": a["stale_rate"],
                 "tone": a.get("tone_mean"), "wall_s": round(time.perf_counter() - t0, 1),
                 "probes": [{"q": d["question"], "score": d["score"],
                             "stale": d["stale_score"], "answer": d["answer"]}
                            for d in a["probe_details"]]}
        summary["conditions"][name] = {k: v for k, v in entry.items() if k != "probes"}
        (DIR / f"{name}.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  recall={a['recall_mean']} confusion={a['stale_rate']} tone={a.get('tone_mean')}", flush=True)
    write_report(summary)
    print("готово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Thread-coherence: удержание нити в меандрирующем диалоге",
        "",
        f"Диалог: {s['turns']} ходов, {s['n_probes']} проб. RAG центрированный, mid-слой, окно 2.",
        "Главная нить: Хельсинки/14 марта/HG-77120/Барсик. Ответвления конфузят: Берлин/Адлон/20 марта/Мурзик.",
        "**recall** — верное значение нити; **confusion** — значение тангента в ответе (потеря нити); **tone** — тема.",
        "",
        "| Форма латента | recall | confusion (stale) | tone |",
        "|---|---|---|---|",
    ]
    for name, c in s["conditions"].items():
        tone = f"{c['tone']:.3f}" if c.get("tone") is not None else "—"
        lines.append(f"| {name} | {c['recall']:.3f} | {c['confusion_stale']:.3f} | {tone} |")
    lines += ["", "v3_clean = чистый центрированный RAG без латента (mid); v3_clean_last = он же на last (как в статье);",
              "additive_v1 = memory=True (аддитивный hook); kv_v4 = memory_kv=True (K/V-инъекция). Все на окне 2.",
              "Ниже confusion = лучше удержание нити. Данные: `<cond>.json` (ответы на пробы).", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Interlocutor-тест: живая память удерживает ГЛОБАЛЬНУЮ ТЕМУ диалога без контекста.

Исходное назначение механизма слияний (battery): быть собеседником — держать
тон/направление диалога из ограниченного буфера, а не recall фактов. Здесь
строгая проверка этой цели: topic continuation (выравнивание ответов на
фиксированный эталон глобальной темы), а не fact-confusion.

3 диалога с рекуррентной глобальной темой + дрейфом подтем. window=2.
Условия изолируют вклад живого merge-gist:
* no_memory    — только окно;
* rag_plain    — окно + plain RAG (без gist);
* rag_gist     — окно + RAG + живой merge-gist (ИСХОДНЫЙ механизм);
* centered_rag — окно + центрированный RAG (v3);
* full_context — полный контекст (upper bound).

Метрика: topic_continuation = mean cos(answer_emb, global_topic_emb) по ходам
(фиксированный эталон, сравнимо между условиями) + tone_alignment.
Гипотеза: rag_gist > rag_plain по удержанию глобальной темы (назначение механизма).

Выход: results/interlocutor/{summary.json,REPORT.md,<dlg>_<cond>.json}
Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/interlocutor_test.py
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
from tmem.qa import DialogSession  # noqa: E402
from tmem.semantic import embed_utterance, tone_alignment  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "interlocutor"

# (name, global_topic, turns) — глобальная тема рекуррентна, подтемы уводят
DIALOGUES = [
    ("trip", "планирование поездки в горы на выходные", [
        "Привет! Помоги спланировать поездку в горы на выходные, это наша главная цель.",
        "Хочу именно в горы, на природу, отдохнуть от города.",
        "Кстати, я недавно читал про историю альпинизма, интересно как снаряжение менялось.",
        "А ещё вспомнил про свой первый рюкзак, он был совсем неудобный.",
        "Но вернёмся к поездке в горы — что взять из снаряжения?",
        "Погода в горах переменчивая, надо учесть.",
        "Я вообще люблю фотографировать пейзажи, может камеру взять.",
        "Помню как в прошлый раз снимал закат, красиво получилось.",
        "Так что по маршруту для нашей горной поездки?",
        "Надо рассчитать время чтобы успеть светлое время суток.",
        "Кстати про время, я часы новые купил, с альтиметром.",
        "Они считают высоту, для гор полезно.",
        "Вернёмся к планированию поездки в горы на выходные.",
        "Где остановиться — палатка или база?",
        "Я склоняюсь к палатке, ближе к природе.",
        "А что по еде в горы брать?",
        "Я тут рецепт трейл-микса освоил, орехи с сухофруктами.",
        "Вообще готовка это отдельная тема, я люблю экспериментировать.",
        "Но для горной поездки нужна простая еда.",
        "Давай финализируем план поездки в горы на выходные.",
        "Составь список что взять и что сделать.",
        "Главное не забыть про безопасность в горах.",
        "Я аптечку соберу обязательно.",
        "Итак, наша поездка в горы на выходные почти спланирована.",
        "Спасибо, что помог сфокусироваться на планировании поездки.",
    ]),
    ("guitar", "обучение игре на гитаре с нуля", [
        "Привет! Я решил научиться играть на гитаре с нуля, помоги мне в этом.",
        "Моя цель именно научиться играть, а не просто купить инструмент.",
        "Кстати, я вчера смотрел концерт, гитарист играл фламенко, завораживает.",
        "Вообще музыка это огромная тема, я люблю джаз.",
        "Но мой фокус — обучение игре на гитаре с нуля.",
        "С чего начать новичку?",
        "Я купил акустику, струны металлические.",
        "Пальцы болят после первых попыток, это нормально?",
        "А ещё я читал про историю гитары, как она из Испании пошла.",
        "Интересно как разные жанры используют гитару.",
        "Вернёмся к моему обучению игре на гитаре.",
        "Какие первые аккорды учить?",
        "Я пробовал Am и E, получается медленно.",
        "Кстати про музыку, я ещё на пианино немного играл в детстве.",
        "Но сейчас хочу именно гитару освоить.",
        "Как тренировать смену аккордов?",
        "Сколько practice в день нужно?",
        "Я могу уделять полчаса каждый вечер.",
        "Давай составим план обучения игре на гитаре с нуля.",
        "Какие песни подходят для новичка?",
        "Хочу сыграть что-то простое но красивое.",
        "Главное регулярность в обучении гитаре.",
        "Итак, мой план обучения игре на гитаре с нуля готов.",
        "Спасибо что помог сфокусироваться на обучении гитаре.",
    ]),
    ("garden", "создание овощного сада на участке", [
        "Привет! Хочу создать овощной сад на участке, помоги спланировать.",
        "Моя цель именно овощной сад, чтобы свои овощи.",
        "Кстати, я читал про японские сады камней, философия интересная.",
        "Вообще ландшафтный дизайн это искусство.",
        "Но мой фокус — овощной сад на участке.",
        "С чего начать планирование сада?",
        "У меня участок шесть соток, солнце с утра.",
        "Я думаю про грядки с помидорами и огурцами.",
        "А ещё вспомнил как у бабушки в деревне был сад, запах помидоров.",
        "Детство в деревне это отдельная тема.",
        "Вернёмся к созданию овощного сада.",
        "Какая почва нужна для овощей?",
        "Я тест почвы сделал, слабокислая.",
        "Кстати про растения, я комнатные цветы люблю, фикусы.",
        "Но сейчас фокус на овощном саде.",
        "Как расположить грядки чтобы удобно было?",
        "Нужно продумать полив.",
        "Я капельный полив хочу сделать.",
        "Давай составим план создания овощного сада на участке.",
        "Что сажать в этом сезоне?",
        "Хочу успеть вырастить урожай.",
        "Главное спланировать овощной сад правильно.",
        "Итак, план овощного сада на участке почти готов.",
        "Спасибо что помог сфокусироваться на создании сада.",
    ]),
]

CONDITIONS = {
    "no_memory": dict(context="window", rag=False, memory=False),
    "rag_plain": dict(context="window", rag=True, rag_centered=False, rag_lambda=0.0, memory=False),
    "rag_gist": dict(context="window", rag=True, rag_centered=False, rag_lambda=0.5,
                     memory=True, memory_merge_k=0.25, memory_crit=0.95, memory_window_crit=0.5),
    "centered_rag": dict(context="window", rag=True, rag_centered=True, rag_lambda=0.0, memory=False),
    "full_context": dict(context="full", rag=False, memory=False),
}


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    summary = {"model": "qwen35-4b", "device": device, "window": 2,
               "n_dialogues": len(DIALOGUES), "conditions": {}}
    # фиксированные эталоны глобальной темы
    refs = {name: embed_utterance(bundle, topic, layer="mean") for name, topic, _ in DIALOGUES}
    refs = {k: torch.nn.functional.normalize(v, dim=0) for k, v in refs.items()}

    per_cond = {c: [] for c in CONDITIONS}
    for name, topic, turns in DIALOGUES:
        session = DialogSession(name=name, opening=turns, fillers=[], probes=[])
        ref = refs[name]
        for cond, cfg in CONDITIONS.items():
            strat = Strategy(f"{name}_{cond}", window=2, **cfg)
            t0 = time.perf_counter()
            run = run_session(bundle, session, strat, max_new_tokens=48, seed=0)
            # topic continuation: cos(answer, fixed global topic) по ходам
            aligns = []
            for t in run.turns:
                if t.answer_text.strip():
                    a = torch.nn.functional.normalize(embed_utterance(bundle, t.answer_text, layer="mean"), dim=0)
                    aligns.append(float(a @ ref))
            tc = sum(aligns) / len(aligns) if aligns else 0.0
            tones = [t.tone_score for t in run.turns if t.tone_score is not None]
            tone = sum(tones) / len(tones) if tones else None
            entry = {"dialogue": name, "topic_continuation": round(tc, 4),
                     "tone": round(tone, 4) if tone is not None else None,
                     "wall_s": round(time.perf_counter() - t0, 1)}
            per_cond[cond].append(entry)
            (DIR / f"{name}_{cond}.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  {name}/{cond}: topic_cont={tc:.4f} tone={tone} ({entry['wall_s']}s)", flush=True)
    for cond, entries in per_cond.items():
        tcs = [e["topic_continuation"] for e in entries]
        summary["conditions"][cond] = {
            "topic_continuation_mean": round(sum(tcs) / len(tcs), 4),
            "per_dialogue": {e["dialogue"]: e["topic_continuation"] for e in entries},
            "tone_mean": round(sum(e["tone"] for e in entries if e["tone"] is not None) /
                               max(1, sum(e["tone"] is not None for e in entries)), 4)}
    (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = ["# Interlocutor: удержание ГЛОБАЛЬНОЙ ТЕМЫ диалога без полного контекста", "",
             f"window={s['window']}, {s['n_dialogues']} диалога. Метрика: topic_continuation",
             "(cos ответа на фиксированный эталон глобальной темы, сравнимо между условиями).",
             "Исходное назначение механизма слияний — собеседник, держащий направление.", "",
             "| Условие | topic_continuation | tone | по диалогам |", "|---|---|---|---|"]
    for cond, c in s["conditions"].items():
        pd = ", ".join(f"{k}={v}" for k, v in c["per_dialogue"].items())
        lines.append(f"| {cond} | {c['topic_continuation_mean']} | {c['tone_mean']} | {pd} |")
    lines += ["", "rag_gist vs rag_plain = вклад живого merge-gist в удержание глобальной темы.",
              "centered_rag = v3 альтернатива; full_context = upper bound. Данные: `<dlg>_<cond>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

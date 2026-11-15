"""Генерация СЛЕПОГО качественного теста (заказ 15.11).

Ценность живой памяти — «человеческий вид»/живость/правдоподобие диалога, а не
строгое topic-соответствие (которое геймится роботом). Оценка вслепую: реплики
модели под разными условиями анонимизируются (метки A–E, shuffle на точке),
соответствие метка→условие хранится ОТДЕЛЬНО (mapping.json) и скрыто от оценщиков.

3 диалога × 5 условий. На точках (ходы 10, 18) — контекст (scripted user turns)
+ 5 реплик-кандидатов A–E. Оценщики (пользователь + сабагенты) ранжируют вслепую
по живости/человечности/правдоподобию. Потом вскрытие mapping и агрегация.

Выход: results/blind_eval/blind_eval.md (для оценщиков), mapping.json (скрыто).
Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/blind_eval_gen.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from interlocutor_test import CONDITIONS, DIALOGUES  # noqa: E402
from tmem.dialogue import Strategy, run_session  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.qa import DialogSession  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "blind_eval"
EVAL_TURNS = [10, 18]
LABELS = ["A", "B", "C", "D", "E"]
RNG_SEED = 20261115


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    cond_names = list(CONDITIONS)

    # генерация полных диалогов под каждым условием
    replies = {}   # (dlg, cond) -> [answer per turn]
    for name, topic, turns in DIALOGUES:
        session = DialogSession(name=name, opening=turns, fillers=[], probes=[])
        for cond, cfg in CONDITIONS.items():
            strat = Strategy(f"{name}_{cond}", window=2, **cfg)
            t0 = time.perf_counter()
            run = run_session(bundle, session, strat, max_new_tokens=64, seed=0)
            replies[(name, cond)] = [t.answer_text for t in run.turns]
            print(f"  {name}/{cond}: {len(run.turns)} ходов ({time.perf_counter()-t0:.0f}s)", flush=True)

    # слепые точки
    rng = random.Random(RNG_SEED)
    items, mapping = [], {}
    item_id = 0
    for name, topic, turns in DIALOGUES:
        for ti in EVAL_TURNS:
            if ti >= len(turns):
                continue
            item_id += 1
            iid = f"item{item_id:02d}"
            cand = {cond: replies[(name, cond)][ti] for cond in cond_names}
            shuffled = cond_names[:]
            rng.shuffle(shuffled)
            label_of = {LABELS[k]: shuffled[k] for k in range(len(shuffled))}
            mapping[iid] = {"dialogue": name, "turn": ti, "labels": label_of}
            ctx = turns[max(0, ti - 2):ti]
            items.append({"iid": iid, "dialogue": name, "turn": ti,
                          "context": ctx, "user": turns[ti],
                          "candidates": {LABELS[k]: cand[shuffled[k]] for k in range(len(shuffled))}})

    # blind_eval.md — БЕЗ имён условий
    lines = ["# Слепая оценка диалога: живость / человечность / правдоподобие", "",
             "Оцени реплики вслепую. На каждой точке дан контекст (реплики пользователя) и",
             "5 вариантов ответа модели (A–E). Ранжируй их от САМОГО живого/человечного/правдоподобного",
             "к наименее (или оцени каждый 1–5). НЕ пытайся угадать условие — оценивай качество общения.",
             "Критерии: естественность, живость, уместность, отсутствие «роботизированности»/зловещей долины.", ""]
    for it in items:
        lines += [f"## {it['iid']} (диалог: {it['dialogue']}, ход {it['turn']})", "",
                  "**Контекст (реплики пользователя):**"]
        for c in it["context"]:
            lines.append(f"> {c}")
        lines += ["", f"**Пользователь:** {it['user']}", "", "**Варианты ответа модели:**", ""]
        for lab in LABELS:
            lines.append(f"**{lab}.** {it['candidates'][lab]}")
            lines.append("")
        lines += ["---", ""]
    (DIR / "blind_eval.md").write_text("\n".join(lines), encoding="utf-8")
    # mapping.json — ОТДЕЛЬНО, скрыто от оценщиков
    (DIR / "mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nготово: {len(items)} слепых точек, {len(cond_names)} условий", flush=True)
    print(f"blind_eval.md (для оценки) + mapping.json (скрыто): {DIR}", flush=True)


if __name__ == "__main__":
    main()

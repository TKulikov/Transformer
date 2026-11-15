"""Memory-contextualized embeddings (заказ 15.11): эмбеддинг в контексте живой памяти.

Тезис автора: использовать активации TTL-банка для СОЗДАНИЯ и КОНТЕКСТУАЛИЗАЦИИ
эмбеддингов — при кодировании текста инжектировать живые слоты банка как K/V в
attention mid-слоёв, так что представление рождается в контексте текущей живой
памяти (не запрос кондиционируется и не генерация, а САМ эмбеддинг).

Это третий механизм, отличный от gist→запрос и K/V→генерация (те не дали
прироста). Здесь банк читается+пишется при кодировании каждого элемента
(живая история): эмбеддинг функции i контекстуализирован памятью функций 1..i-1.

Условия (taskq_large, semtree mean/tau=0.6, k=6):
* standard     — обычные эмбеддинги (базлайн 9/10);
* memctx_keys  — K/V-инжект живой памяти при кодировании КЛЮЧЕЙ (запрос обычный);
* memctx_both  — контекстуализированы и ключи, и запрос.

Метрики: reuse / retrieval hits / def hits. Выход:
results/memctx/{summary.json,REPORT.md,<cond>.json}
Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/memctx_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from code_search_retest import generate, score_entry  # noqa: E402
from taskq_large_eval import PROJECT, TASKS, project_context  # noqa: E402
from tmem.hiermem import build_project_tree  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.semtree import SemTreeBank  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "memctx"
K, K_CLUSTERS, SPACE, TAU = 6, 3, "mean", 0.6
CONDITIONS = [
    ("standard", False, True),
    ("memctx_keys", True, False),
    ("memctx_both", True, True),
]


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    nodes = build_project_tree(PROJECT)
    ctx = project_context()
    print(f"проект: {len(nodes)} узлов", flush=True)

    summary = {"model": "qwen35-4b", "device": device, "space": SPACE, "tau": TAU,
               "k": K, "conditions": {}}
    for cond, memctx, memctx_query in CONDITIONS:
        bank = SemTreeBank(nodes, space=SPACE, tau=TAU, device=bundle.device,
                           memory_context=memctx, memctx_query=memctx_query)
        t0 = time.perf_counter()
        bank.ingest(bundle)
        ingest_s = time.perf_counter() - t0
        slots = bank.stats.get("memctx_slots", 0)
        print(f"\n=== {cond} === ингестия {ingest_s:.0f}s, живых слотов {slots}", flush=True)
        entries = []
        for task in TASKS:
            res, diag = bank.retrieve(bundle, task["user"], k_clusters=K_CLUSTERS, k=K)
            notes = [n.text for n, _, _ in res]
            answer = generate(bundle, ctx, notes, task["user"])
            entry = score_entry(task, answer, notes)
            entry["def_hits"] = {n: any(f"def {n}" in note for note in notes)
                                 for n in task["expected"]}
            entry["clusters_picked"] = diag
            entries.append(entry)
            print(f"  {task['id']}: reuse={entry['reuse']} def={entry['def_hits']}", flush=True)
        reuse = sum(sum(e["reuse"].values()) for e in entries)
        hits = sum(sum(e["retrieval_hit"].values()) for e in entries)
        defs = sum(sum(e["def_hits"].values()) for e in entries)
        summary["conditions"][cond] = {"reuse": reuse, "hits": hits, "def_hits": defs,
                                       "ingest_s": round(ingest_s, 1), "slots": slots,
                                       "entries": entries}
        (DIR / f"{cond}.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Memory-contextualized embeddings: эмбеддинг в контексте живой памяти",
        "",
        "При кодировании текста живые слоты TTL-банка инжектируются как K/V в mid-attention,",
        "т.е. эмбеддинг рождается в контексте текущей живой памяти (банк читается+пишется).",
        f"semtree space={s['space']}, tau={s['tau']}, k={s['k']}.",
        "",
        "| Условие | Reuse /10 | Hits /10 | Def hits /10 | Живых слотов | Ингестия, с |",
        "|---|---|---|---|---|---|",
    ]
    for cond, c in s["conditions"].items():
        lines.append(f"| {cond} | {c['reuse']} | {c['hits']} | {c['def_hits']} | {c['slots']} | {c['ingest_s']} |")
    lines += ["", "standard — обычные эмбеддинги; memctx_keys — контекстуализированы ключи;",
              "memctx_both — ключи и запрос. Def hits — определение хелпера в notes.",
              "Данные: `<cond>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

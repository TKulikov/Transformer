"""Глубокое семантическое дерево на taskq_large (заказ 15.11, ночь).

Проверка: группа — тоже узел; похожие центроиды объединяются рекурсивно в
более глубокое дерево. Поиск по нему — beam-спуск с пере-резидуализацией
запроса на каждом уровне (механизм descent, застрявший на файловом дереве,
теперь на семантическом). Изолируем эффект глубины/структуры: space=mean
(дискриминативный лидер ретеста), tau=0.6 фиксированы.

Условия (те же 5 задач/генерация):
* deep1_flat    — двухуровневое дерево, плоский поиск по группам (контроль);
* deep1_descent — beam-спуск по двухуровневому;
* deep2_descent — рекурсивное объединение центроидов, глубина 2, спуск;
* deep3_descent — глубина 3, спуск;
* flat_legacy   — референс-лидер (8/10 дважды).

Диагностика: реальная глубина/число узлов дерева (углубляется ли вообще на
этих данных), выбранные группы, def-хиты. Ингестия одна (recluster без форвардов).
Выход: results/chunk_code/semtree_deep/{summary.json,REPORT.md,<cond>.json}

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/semtree_deep_eval.py
Очередь: scripts/queue_semtree_deep.sh (ждёт окончания semtree_eval).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from code_search_retest import build_flat_bank, generate, score_entry  # noqa: E402
from taskq_large_eval import PROJECT, TASKS, project_context  # noqa: E402
from tmem.hiermem import build_project_tree  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.semtree import SemTreeBank  # noqa: E402
from tmem.semantic import embed_utterance  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "chunk_code" / "semtree_deep"
K, K_CLUSTERS, SPACE, TAU = 6, 3, "mean", 0.6
CONDITIONS = [
    ("deep1_flat", 1, "flat"),
    ("deep1_descent", 1, "descent"),
    ("deep2_descent", 2, "descent"),
    ("deep3_descent", 3, "descent"),
]


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    files = [p.read_text(encoding="utf-8") for p in sorted(PROJECT.rglob("*.py"))]
    nodes = build_project_tree(PROJECT)
    ctx = project_context()

    bank = SemTreeBank(nodes, space=SPACE, tau=TAU, device=bundle.device)
    t0 = time.perf_counter()
    bank.ingest(bundle)
    ingest_s = time.perf_counter() - t0
    print(f"semtree ингестия: {bank.stats} за {ingest_s:.0f}s", flush=True)

    flat_legacy, flat_s = build_flat_bank(bundle, files, "mean", "mid")
    print(f"flat_legacy: {len(flat_legacy.texts)} чанков за {flat_s:.0f}s", flush=True)

    summary = {"model": "qwen35-4b", "device": device, "project": "taskq_large",
               "space": SPACE, "tau": TAU, "k": K, "k_clusters": K_CLUSTERS,
               "ingest_s": {"semtree": round(ingest_s, 1), "flat_legacy": round(flat_s, 1)},
               "conditions": {}}

    def run_cond(cond: str, retrieve_fn, tree_stats=None) -> None:
        print(f"\n=== {cond} === {tree_stats or ''}", flush=True)
        t0 = time.perf_counter()
        entries = []
        for task in TASKS:
            notes, diag = retrieve_fn(task["user"])
            answer = generate(bundle, ctx, notes, task["user"])
            entry = score_entry(task, answer, notes)
            entry["clusters_picked"] = diag
            entry["def_hits"] = {n: any(f"def {n}" in note for note in notes)
                                 for n in task["expected"]}
            entries.append(entry)
            print(f"  {task['id']}: reuse={entry['reuse']} hits={entry['retrieval_hit']} "
                  f"def={entry['def_hits']}", flush=True)
        summary["conditions"][cond] = {"wall_s": round(time.perf_counter() - t0, 1),
                                       "tree_stats": tree_stats, "entries": entries}
        (DIR / f"{cond}.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    for cond, depth, mode in CONDITIONS:
        bank.recluster(SPACE, TAU, max_depth=depth)
        ts = {k: bank.stats[k] for k in ("clusters", "singletons", "max_cluster", "depth", "nodes")}

        def retrieve(q, _b=bank, _m=mode):
            res, diag = _b.retrieve(bundle, q, k_clusters=K_CLUSTERS, k=K, mode=_m)
            return [n.text for n, _, _ in res], diag

        run_cond(cond, retrieve, ts)

    def retrieve_flat(q):
        return [t for t, _, _ in flat_legacy.retrieve(embed_utterance(bundle, q, layer="last"),
                                                      k=K, score="key")], []

    run_cond("flat_legacy", retrieve_flat)
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Глубокое семантическое дерево (taskq_large): группы объединяются рекурсивно",
        "",
        f"space={s['space']}, tau={s['tau']}, k={s['k']}, beam={s['k_clusters']}.",
        "Группа — узел; похожие центроиды сливаются под общего родителя (дендрограмма).",
        "descent — beam-спуск с пере-резидуализацией запроса на каждом уровне.",
        f"Ингестия (с): {s['ingest_s']}.",
        "",
        "| Условие | Глубина | Узлов | Групп | Reuse /10 | Hits /10 | Def hits /10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for cond, c in s["conditions"].items():
        ents = c["entries"]
        reuse = sum(sum(e["reuse"].values()) for e in ents)
        hits = sum(sum(e["retrieval_hit"].values()) for e in ents)
        defs = sum(sum(e.get("def_hits", {}).values()) for e in ents)
        ts = c.get("tree_stats") or {}
        lines.append(f"| {cond} | {ts.get('depth', '—')} | {ts.get('nodes', '—')} "
                     f"| {ts.get('clusters', '—')} | {reuse} | {hits} | {defs} |")
    lines += ["", "Def hits — определение хелпера в notes (не только вызов).",
              "Глубина 2 = корень→группы→листья; если центроиды не похожи, дерево не углубляется.",
              "Данные: `summary.json`, `<cond>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

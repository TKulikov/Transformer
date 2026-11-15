"""Семантическое дерево на taskq_large (заказ 15.11, ночь).

Проверка предложения: модель сама собирает дерево связей — онлайн-группы
похожих функций (leader-кластеризация в deviation-пространстве), ключи
листьев = отклонение от среднего группы; поиск: запрос → top групп →
residual-поиск внутри. Лечит слепоту file-first к grab-bag utils (fileres).

Условия (те же 5 задач/генерация, что search_retest/fileres):
* semtree_mid_t40/t60/t80  — mid-ключи, tau=0.4/0.6/0.8;
* semtree_mean_t40/t60/t80 — mean-ключи (дискриминативные), те же tau;
* flat_legacy — референс-лидер (8/10 дважды).

k_clusters=3, k=6 заметок. Диагностика: размеры групп, какие группы выбраны
на задачу, hits по def vs по вызову. Ингестия одна (recluster без форвардов).
Выход: results/chunk_code/semtree/{summary.json,REPORT.md,<cond>.json}

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/semtree_eval.py
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
DIR = ROOT / "results" / "chunk_code" / "semtree"
K, K_CLUSTERS = 6, 3
CONDITIONS = [(f"semtree_{space}_t{int(tau * 100)}", space, tau)
              for space in ("mid", "mean") for tau in (0.4, 0.6, 0.8)]
HELPERS = ["retry", "backoff_delay", "parse_duration", "clamp", "dedupe",
           "chunked", "slugify", "compute_total", "summarize", "domain_metric"]


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    files = [p.read_text(encoding="utf-8") for p in sorted(PROJECT.rglob("*.py"))]
    nodes = build_project_tree(PROJECT)
    ctx = project_context()

    bank = SemTreeBank(nodes, device=bundle.device)
    t0 = time.perf_counter()
    bank.ingest(bundle)
    ingest_s = time.perf_counter() - t0
    print(f"semtree ингестия: {bank.stats} за {ingest_s:.0f}s", flush=True)

    # распределение сходств в deviation-пространстве (ориентир для tau)
    sim_stats = {}
    for space in ("mid", "mean"):
        dev = bank.dev if bank.space == space else None
        bank.recluster(space, 0.6)
        sims = bank.dev @ bank.dev.T
        off = sims[~torch.eye(sims.shape[0], dtype=torch.bool, device=sims.device)]
        qs = torch.quantile(off.float(), torch.tensor([0.5, 0.9, 0.99], device=off.device))
        sim_stats[space] = {f"p{int(p * 100)}": round(float(v), 3)
                            for p, v in zip((0.5, 0.9, 0.99), qs)}
    print(f"сходства deviation: {sim_stats}", flush=True)

    flat_legacy, flat_s = build_flat_bank(bundle, files, "mean", "mid")
    print(f"flat_legacy: {len(flat_legacy.texts)} чанков за {flat_s:.0f}s", flush=True)

    summary = {"model": "qwen35-4b", "device": device, "project": "taskq_large",
               "k": K, "k_clusters": K_CLUSTERS, "sim_percentiles": sim_stats,
               "ingest_s": {"semtree": round(ingest_s, 1), "flat_legacy": round(flat_s, 1)},
               "conditions": {}}

    def run_cond(cond: str, retrieve_fn, cluster_stats=None) -> None:
        print(f"\n=== {cond} === {cluster_stats or ''}", flush=True)
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
                  f"def={entry['def_hits']} groups={diag}", flush=True)
        summary["conditions"][cond] = {"wall_s": round(time.perf_counter() - t0, 1),
                                       "cluster_stats": cluster_stats, "entries": entries}
        (DIR / f"{cond}.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    for cond, space, tau in CONDITIONS:
        bank.recluster(space, tau)
        cs = {k: bank.stats[k] for k in ("clusters", "singletons", "max_cluster")}

        def retrieve(q, _b=bank):
            res, diag = _b.retrieve(bundle, q, k_clusters=K_CLUSTERS, k=K)
            return [n.text for n, _, _ in res], diag

        run_cond(cond, retrieve, cs)

    def retrieve_flat(q):
        return [t for t, _, _ in flat_legacy.retrieve(embed_utterance(bundle, q, layer="last"),
                                                      k=K, score="key")], []

    run_cond("flat_legacy", retrieve_flat)
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Семантическое дерево связей (taskq_large): группы по смыслу, не по файлам",
        "",
        "Онлайн leader-кластеризация в deviation-пространстве; ключ листа = отклонение",
        "от среднего группы; поиск: top-3 группы по центроиду → residual-поиск внутри.",
        f"k={s['k']} заметок. Сходства deviation (percentiles): {s['sim_percentiles']}.",
        f"Ингестия (с): {s['ingest_s']}.",
        "",
        "| Условие | Групп | Одиночек | Макс группа | Reuse /10 | Hits /10 | Def hits /10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for cond, c in s["conditions"].items():
        ents = c["entries"]
        reuse = sum(sum(e["reuse"].values()) for e in ents)
        hits = sum(sum(e["retrieval_hit"].values()) for e in ents)
        defs = sum(sum(e.get("def_hits", {}).values()) for e in ents)
        cs = c.get("cluster_stats") or {}
        lines.append(f"| {cond} | {cs.get('clusters', '—')} | {cs.get('singletons', '—')} "
                     f"| {cs.get('max_cluster', '—')} | {reuse} | {hits} | {defs} |")
    lines += ["", "## Выбранные группы (stage 1, top-3 имени)", "",
              "| Условие | " + " | ".join(t["id"] for t in TASKS) + " |",
              "|---|" + "---|" * len(TASKS)]
    for cond, c in s["conditions"].items():
        cells = []
        for e in c["entries"]:
            gs = e.get("clusters_picked") or []
            cells.append("; ".join(f"[{g['size']}] " + "/".join(g["members"][:2]) for g in gs[:2]) or "—")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["", "Def hits — определение хелпера в notes (не только вызов); flat_legacy — референс.",
              "Данные: `summary.json`, `<cond>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

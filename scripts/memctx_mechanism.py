"""Механика выигрыша memctx на ИСХОДНОМ порядке (заказ 15.11).

Verification тестировал 8 случайных порядков; исходный порядок дерева (давший
+1 hit/+1 def в memctx_eval) не характеризован. Разбираем МЕХАНИКУ: что именно
memctx_keys извлекает, чего не извлекает standard, и почему контекстуализация
живой памятью сдвигает эмбеддинг/кластер нужной функции.

Без генерации — анализ retrieval (def-hit = определение в notes). Для каждой
задачи: какие def найдены std vs mem, различия; для различающихся функций —
кластер, cos(std_key, mem_key), residual-score к запросу.

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/memctx_mechanism.py
Выход: results/memctx_mechanism/{REPORT.md,detail.json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from taskq_large_eval import PROJECT, TASKS  # noqa: E402
from tmem.hiermem import build_project_tree  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.semtree import SemTreeBank  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "memctx_mechanism"
K, K_CLUSTERS, SPACE, TAU = 6, 3, "mean", 0.6
HELPERS = ["retry", "backoff_delay", "parse_duration", "clamp", "dedupe",
           "chunked", "slugify", "compute_total", "summarize", "domain_metric"]


def cluster_map(bank):
    m = {}
    for j, cl in enumerate(bank.clusters):
        for i in cl["members"]:
            m[i] = j
    return m


def idx_of(bank, name):
    for i, n in enumerate(bank.funcs):
        if n.name == name:
            return i
    return None


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    nodes = build_project_tree(PROJECT)   # ИСХОДНЫЙ порядок, без shuffle
    std = SemTreeBank(nodes, space=SPACE, tau=TAU, device=bundle.device, memory_context=False)
    std.ingest(bundle)
    mem = SemTreeBank(nodes, space=SPACE, tau=TAU, device=bundle.device,
                      memory_context=True, memctx_query=False)
    mem.ingest(bundle)
    cmap_s, cmap_m = cluster_map(std), cluster_map(mem)
    print(f"кластеров: std={len(std.clusters)} mem={len(mem.clusters)}; "
          f"живых слотов mem={mem.stats.get('memctx_slots')}", flush=True)

    detail = {"n_clusters_std": len(std.clusters), "n_clusters_mem": len(mem.clusters),
              "tasks": []}
    for task in TASKS:
        rs, _ = std.retrieve(bundle, task["user"], k_clusters=K_CLUSTERS, k=K)
        rm, _ = mem.retrieve(bundle, task["user"], k_clusters=K_CLUSTERS, k=K)
        notes_s = [n.text for n, _, _ in rs]
        notes_m = [n.text for n, _, _ in rm]
        defs_s = {h: any(f"def {h}" in t for t in notes_s) for h in task["expected"]}
        defs_m = {h: any(f"def {h}" in t for t in notes_m) for h in task["expected"]}
        names_s = [n.name for n, _, _ in rs]
        names_m = [n.name for n, _, _ in rm]
        gained = [h for h in task["expected"] if defs_m[h] and not defs_s[h]]
        lost = [h for h in task["expected"] if defs_s[h] and not defs_m[h]]
        entry = {"task": task["id"], "defs_std": defs_s, "defs_mem": defs_m,
                 "gained": gained, "lost": lost,
                 "top_std": names_s, "top_mem": names_m, "mechanism": []}
        # механика для gained/различающихся функций
        for h in set(gained + lost):
            i_s, i_m = idx_of(std, h), idx_of(mem, h)
            if i_s is None or i_m is None:
                continue
            ks = std.raw[SPACE][i_s]; km = mem.raw[SPACE][i_m]
            cos = float(torch.nn.functional.normalize(ks, dim=0) @ torch.nn.functional.normalize(km, dim=0))
            entry["mechanism"].append({
                "helper": h, "cos_key_std_mem": round(cos, 4),
                "cluster_std": cmap_s.get(i_s), "cluster_mem": cmap_m.get(i_m),
                "cluster_size_std": len(std.clusters[cmap_s[i_s]]["members"]) if i_s in cmap_s else None,
                "cluster_size_mem": len(mem.clusters[cmap_m[i_m]]["members"]) if i_m in cmap_m else None,
                "in_top_std": h in names_s, "in_top_mem": h in names_m,
            })
        detail["tasks"].append(entry)
        print(f"\n{task['id']}: gained={gained} lost={lost}", flush=True)
        print(f"  std top: {names_s}", flush=True)
        print(f"  mem top: {names_m}", flush=True)
        for m in entry["mechanism"]:
            print(f"  {m}", flush=True)
    (DIR / "detail.json").write_text(json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(detail)
    print("\nготово:", DIR, flush=True)


def write_report(d: dict) -> None:
    lines = ["# Механика выигрыша memctx (исходный порядок дерева)", "",
             f"Кластеров: std={d['n_clusters_std']}, mem={d['n_clusters_mem']}.", "",
             "| Задача | gained (mem нашёл, std нет) | lost | cos ключа std↔mem | кластер std→mem |",
             "|---|---|---|---|---|"]
    for t in d["tasks"]:
        mech = "; ".join(f"{m['helper']}: cos={m['cos_key_std_mem']} cl{m['cluster_std']}→{m['cluster_mem']}"
                         for m in t["mechanism"]) or "—"
        lines.append(f"| {t['task']} | {', '.join(t['gained']) or '—'} | {', '.join(t['lost']) or '—'} "
                     f"| {mech} |")
    lines += ["", "cos ключа — насколько контекстуализация живой памятью сдвинула эмбеддинг функции;",
              "кластер std→mem — сменила ли функция группу. Данные: detail.json.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Диверсификация как робастный механизм (заказ 15.11).

Механика выигрыша memctx на исходном порядке: крошечный сдвиг эмбеддинга
(cos 0.9987) поднял retry ВЫШЕ клонов timeout_ms (5/6 слотов) в residual-
ранжировании. Выигрыш реальный, но хрупкий (случайный tie-break). Корень —
flooding клонами. Робастный механизм — явная диверсификация (MMR): разводит
клонов, чтобы погребённые функции всплывали надёжно на ЛЮБОМ порядке.

Гипотеза: diverse >> plain робастно (по сидам); memctx добавляет мало поверх
diverse (его эффект был хрупким прокси диверсификации).

8 порядков (исходный + 7 shuffle), 4 комбинации: std_plain / std_diverse /
mem_plain / mem_diverse. Метрики retrieval (hits / def-hits, без генерации).
Парные дельты: diverse−plain (std), memctx−std (plain и diverse).

Выход: results/memctx_diverse/{summary.json,REPORT.md}
Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/memctx_diverse.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from taskq_large_eval import PROJECT, TASKS  # noqa: E402
from tmem.hiermem import build_project_tree  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.semtree import SemTreeBank  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "memctx_diverse"
K, K_CLUSTERS, SPACE, TAU, LAM_DIV = 6, 3, "mean", 0.6, 0.5
ORDERS = [None] + list(range(7))   # None = исходный порядок дерева


def retrieval_metrics(bank, bundle, diverse: bool) -> dict:
    hits = defs = 0
    per = []
    for task in TASKS:
        res, _ = bank.retrieve(bundle, task["user"], k_clusters=K_CLUSTERS, k=K,
                               diverse=diverse, lam_div=LAM_DIV)
        notes = [n.text for n, _, _ in res]
        h = sum(any((f"def {n}" in t) or (f"{n}(" in t) for t in notes) for n in task["expected"])
        d = sum(any(f"def {n}" in t for t in notes) for n in task["expected"])
        hits += h; defs += d
        per.append({"task": task["id"], "hits": h, "def_hits": d})
    return {"hits": hits, "def_hits": defs, "per": per}


def stats(deltas: list[int]) -> dict:
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    t = mean / (std / math.sqrt(n)) if std > 0 else (float("inf") if mean else 0.0)
    return {"mean": round(mean, 3), "std": round(std, 3), "t": round(t, 3), "df": n - 1,
            "pos": sum(d > 0 for d in deltas), "neg": sum(d < 0 for d in deltas),
            "zero": sum(d == 0 for d in deltas)}


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    nodes = build_project_tree(PROJECT)
    rows = []
    for order in ORDERS:
        t0 = time.perf_counter()
        std = SemTreeBank(nodes, space=SPACE, tau=TAU, device=bundle.device, memory_context=False)
        mem = SemTreeBank(nodes, space=SPACE, tau=TAU, device=bundle.device,
                          memory_context=True, memctx_query=False)
        if order is not None:
            random.Random(order).shuffle(std.funcs)
            random.Random(order).shuffle(mem.funcs)
        std.ingest(bundle)
        mem.ingest(bundle)
        sp = retrieval_metrics(std, bundle, diverse=False)
        sd = retrieval_metrics(std, bundle, diverse=True)
        mp = retrieval_metrics(mem, bundle, diverse=False)
        md = retrieval_metrics(mem, bundle, diverse=True)
        row = {"order": "orig" if order is None else order,
               "std_plain": sp, "std_diverse": sd, "mem_plain": mp, "mem_diverse": md}
        rows.append(row)
        print(f"  order {row['order']}: std_plain h{sp['hits']}/d{sp['def_hits']} "
              f"std_div h{sd['hits']}/d{sd['def_hits']} mem_plain h{mp['hits']}/d{mp['def_hits']} "
              f"mem_div h{md['hits']}/d{md['def_hits']} ({time.perf_counter()-t0:.0f}s)", flush=True)
    summary = {"model": "qwen35-4b", "lam_div": LAM_DIV, "n_orders": len(ORDERS), "rows": rows,
               "diverse_effect_hits": stats([r["std_diverse"]["hits"] - r["std_plain"]["hits"] for r in rows]),
               "diverse_effect_def": stats([r["std_diverse"]["def_hits"] - r["std_plain"]["def_hits"] for r in rows]),
               "memctx_plain_hits": stats([r["mem_plain"]["hits"] - r["std_plain"]["hits"] for r in rows]),
               "memctx_diverse_hits": stats([r["mem_diverse"]["hits"] - r["std_diverse"]["hits"] for r in rows]),
               "means": {c: {m: round(sum(r[c][m] for r in rows) / len(rows), 2) for m in ("hits", "def_hits")}
                         for c in ("std_plain", "std_diverse", "mem_plain", "mem_diverse")}}
    (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = ["# Диверсификация (MMR) как робастный механизм", "",
             f"lam_div={s['lam_div']}, {s['n_orders']} порядков (исходный + shuffle). Метрики retrieval.", "",
             "## Среднее по порядкам", "",
             "| Комбинация | hits /10 | def-hits /10 |", "|---|---|---|"]
    for c, m in s["means"].items():
        lines.append(f"| {c} | {m['hits']} | {m['def_hits']} |")
    lines += ["", "## Парные эффекты (t-крит df=7 → 2.365)", "",
              "| Эффект | mean Δ | std | t | +/−/0 |", "|---|---|---|---|---|"]
    for label, key in [("diverse−plain (hits)", "diverse_effect_hits"),
                       ("diverse−plain (def)", "diverse_effect_def"),
                       ("memctx−std plain (hits)", "memctx_plain_hits"),
                       ("memctx−std diverse (hits)", "memctx_diverse_hits")]:
        st = s[key]
        lines.append(f"| {label} | {st['mean']} | {st['std']} | {st['t']} | {st['pos']}/{st['neg']}/{st['zero']} |")
    lines += ["", "## По порядкам (hits/def-hits)", "",
              "| order | std_plain | std_diverse | mem_plain | mem_diverse |", "|---|---|---|---|---|"]
    for r in s["rows"]:
        cells = [f"{r[c]['hits']}/{r[c]['def_hits']}" for c in ("std_plain", "std_diverse", "mem_plain", "mem_diverse")]
        lines.append(f"| {r['order']} | " + " | ".join(cells) + " |")
    lines += ["", "Гипотеза: diverse >> plain робастно; memctx добавляет мало поверх diverse.",
              "Данные: summary.json.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

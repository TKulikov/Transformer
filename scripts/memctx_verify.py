"""Verification memory-contextualized embeddings (заказ 15.11): статистика + диалог.

После фикса маски (68add5b) memctx_keys дал +1 hit / +1 def-hit на коде (n=1).
Здесь проверяем устойчивость:

Часть A — КОД, N сидов: порядок ингестии shuffle (живая память онлайн-зависима
от порядка — легитимный источник variance). На каждый сид standard и memctx_keys
ингестятся в ОДНОМ порядке (парное сравнение), greedy-оценка 5 задач. Парные
дельты (memctx−standard) по hits/def-hits/reuse → mean±std, t-статистика, знак.

Часть B — ДИАЛОГ (thread-тест): перепроверка условий с ФИКСОМ маски (прежние
K/V-результаты были decode-only, prompt-prefill память не работала). Сравнение
с прежними результатами (results/thread_coherence).

Выход: results/memctx_verify/{summary.json,REPORT.md}
Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/memctx_verify.py
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

from code_search_retest import generate, score_entry  # noqa: E402
from taskq_large_eval import PROJECT, TASKS, project_context  # noqa: E402
from thread_coherence_test import CONDITIONS as DLG_CONDS, build_thread_dialogue  # noqa: E402
from tmem.dialogue import Strategy, run_session  # noqa: E402
from tmem.hiermem import build_project_tree  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.semtree import SemTreeBank  # noqa: E402
from tmem.stress import analyze_session  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "memctx_verify"
K, K_CLUSTERS, SPACE, TAU, N_SEEDS = 6, 3, "mean", 0.6, 8


def code_condition(bundle, nodes, ctx, seed: int, memctx: bool) -> dict:
    bank = SemTreeBank(nodes, space=SPACE, tau=TAU, device=bundle.device,
                       memory_context=memctx, memctx_query=False)
    random.Random(seed).shuffle(bank.funcs)   # общий порядок для парного сравнения
    bank.ingest(bundle)
    reuse = hits = defs = 0
    per_task = []
    for task in TASKS:
        res, _ = bank.retrieve(bundle, task["user"], k_clusters=K_CLUSTERS, k=K)
        notes = [n.text for n, _, _ in res]
        answer = generate(bundle, ctx, notes, task["user"])
        e = score_entry(task, answer, notes)
        r = sum(e["reuse"].values())
        h = sum(e["retrieval_hit"].values())
        d = sum(any(f"def {n}" in note for note in notes) for n in task["expected"])
        reuse += r; hits += h; defs += d
        per_task.append({"task": task["id"], "reuse": r, "hits": h, "def_hits": d})
    return {"reuse": reuse, "hits": hits, "def_hits": defs, "per_task": per_task}


def stats(deltas: list[int]) -> dict:
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    t = mean / (std / math.sqrt(n)) if std > 0 else (float("inf") if mean != 0 else 0.0)
    return {"mean": round(mean, 3), "std": round(std, 3), "t": round(t, 3),
            "df": n - 1, "pos": sum(d > 0 for d in deltas),
            "neg": sum(d < 0 for d in deltas), "zero": sum(d == 0 for d in deltas)}


def part_code(bundle, summary) -> None:
    nodes = build_project_tree(PROJECT)
    ctx = project_context()
    print(f"\n########## ЧАСТЬ A: КОД, {N_SEEDS} сидов ##########", flush=True)
    rows = []
    for seed in range(N_SEEDS):
        t0 = time.perf_counter()
        std = code_condition(bundle, nodes, ctx, seed, memctx=False)
        mem = code_condition(bundle, nodes, ctx, seed, memctx=True)
        row = {"seed": seed, "standard": std, "memctx_keys": mem,
               "d_hits": mem["hits"] - std["hits"],
               "d_def": mem["def_hits"] - std["def_hits"],
               "d_reuse": mem["reuse"] - std["reuse"]}
        rows.append(row)
        print(f"  seed {seed}: std hits={std['hits']} def={std['def_hits']} reuse={std['reuse']} | "
              f"mem hits={mem['hits']} def={mem['def_hits']} reuse={mem['reuse']} | "
              f"Δhits={row['d_hits']} Δdef={row['d_def']} ({time.perf_counter()-t0:.0f}s)", flush=True)
    summary["code"] = {
        "n_seeds": N_SEEDS, "rows": rows,
        "hits_delta": stats([r["d_hits"] for r in rows]),
        "def_delta": stats([r["d_def"] for r in rows]),
        "reuse_delta": stats([r["d_reuse"] for r in rows]),
        "std_mean": {m: round(sum(r["standard"][m] for r in rows) / N_SEEDS, 2)
                     for m in ("reuse", "hits", "def_hits")},
        "mem_mean": {m: round(sum(r["memctx_keys"][m] for r in rows) / N_SEEDS, 2)
                     for m in ("reuse", "hits", "def_hits")},
    }
    (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def part_dialogue(bundle, summary) -> None:
    session = build_thread_dialogue()
    print(f"\n########## ЧАСТЬ B: ДИАЛОГ (фикс маски) ##########", flush=True)
    summary["dialogue"] = {}
    for name, cfg in DLG_CONDS:
        strat = Strategy(name, context="window", window=2, rag=True, rag_lambda=0.0,
                         rag_centered=True, **cfg)
        run = run_session(bundle, session, strat, max_new_tokens=48, seed=0)
        a = analyze_session({name: run})[name]
        summary["dialogue"][name] = {"recall": a["recall_mean"], "confusion": a["stale_rate"],
                                     "tone": a.get("tone_mean"), "kv_injected": run.kv_injected}
        print(f"  {name}: recall={a['recall_mean']} confusion={a['stale_rate']} "
              f"tone={a.get('tone_mean')} kv={run.kv_injected}", flush=True)
    (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    summary = {"model": "qwen35-4b", "device": device, "n_seeds": N_SEEDS}
    part_code(bundle, summary)
    part_dialogue(bundle, summary)
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    c = s["code"]
    lines = [
        "# Verification memory-contextualized embeddings",
        "",
        f"## Часть A — код, {c['n_seeds']} сидов (парное сравнение, общий порядок ингестии)",
        "",
        f"Среднее: standard {c['std_mean']} | memctx_keys {c['mem_mean']}",
        "",
        "| Метрика | mean Δ | std | t | df | +/−/0 |",
        "|---|---|---|---|---|---|",
    ]
    for label, key in [("hits", "hits_delta"), ("def-hits", "def_delta"), ("reuse", "reuse_delta")]:
        st = c[key]
        lines.append(f"| {label} | {st['mean']} | {st['std']} | {st['t']} | {st['df']} "
                     f"| {st['pos']}/{st['neg']}/{st['zero']} |")
    lines += ["", "### По сидам", "",
              "| seed | std hits/def/reuse | mem hits/def/reuse | Δhits | Δdef | Δreuse |",
              "|---|---|---|---|---|---|"]
    for r in c["rows"]:
        st, me = r["standard"], r["memctx_keys"]
        lines.append(f"| {r['seed']} | {st['hits']}/{st['def_hits']}/{st['reuse']} "
                     f"| {me['hits']}/{me['def_hits']}/{me['reuse']} "
                     f"| {r['d_hits']} | {r['d_def']} | {r['d_reuse']} |")
    lines += ["", "## Часть B — диалог (thread-тест, С ФИКСОМ маски)", "",
              "| Условие | recall | confusion | tone | K/V inj |",
              "|---|---|---|---|---|"]
    for name, d in s["dialogue"].items():
        tone = f"{d['tone']:.3f}" if d.get("tone") is not None else "—"
        lines.append(f"| {name} | {d['recall']:.3f} | {d['confusion']:.3f} | {tone} | {d['kv_injected']} |")
    lines += ["", "t-крит (двусторонний, α=0.05): df=7 → 2.365. |t|>крит → значимо.",
              "Часть B сравнить с прежними results/thread_coherence (до фикса маски).",
              "Данные: summary.json.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

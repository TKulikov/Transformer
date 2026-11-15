"""Живая память как ядро: абляция TTL-динамики и gist-readout (заказ 15.11).

Тезис: ядро системы — ЖИВАЯ память темы (TTL refresh/decay/eviction + merge,
питаемая mid-активациями, обновляемая запросами), а не K/V-инъекция и не
дерево сами по себе. Gist — лишь один readout; живая память работает и без
него. Проверяем эмпирически, изолируя:
  * TTL-динамику (decay on/off = живая vs статическая память);
  * gist-readout (alpha/rag_lambda = 0 / 0.5 / 1.0).

Часть A — КОД (taskq_large, semtree mean/tau=0.6): банк влияет на поиск
через gist, поэтому alpha=0 = банк не влияет (базлайн). living vs static при
alpha>0 изолирует эффект TTL-динамики на gist.
  nobank | static_a05 | living_a05 | static_a10 | living_a10

Часть B — ДИАЛОГ (thread-тест, меандр с дрейфом темы): банк входит в
генерацию напрямую (K/V), поэтому lambda=0 тестирует живую память БЕЗ gist.
  v3_nomem | kv_living_a0 | kv_static_a0 | kv_living_a05 | kv_living_a10
Метрики: recall, confusion (потеря нити), tone.

Выход: results/living_memory/{summary.json,REPORT.md,code_<cond>.json,dlg_<cond>.json}
Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/living_memory_eval.py
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
from thread_coherence_test import build_thread_dialogue  # noqa: E402
from tmem.dialogue import Strategy, run_session  # noqa: E402
from tmem.hiermem import build_project_tree  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.semtree import SemTreeBank  # noqa: E402
from tmem.stress import analyze_session  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "living_memory"
K, K_CLUSTERS, SPACE, TAU = 6, 3, "mean", 0.6
CODE_CONDS = [
    ("nobank", False, True, 0.0),
    ("static_a05", True, False, 0.5),
    ("living_a05", True, True, 0.5),
    ("static_a10", True, False, 1.0),
    ("living_a10", True, True, 1.0),
]
MEM = dict(memory_window_crit=0.5, memory_crit=0.95, memory_merge_k=0.25, kv_slots=32)
DLG_CONDS = [
    ("v3_nomem", dict(memory=False, memory_kv=False, rag_lambda=0.0)),
    ("kv_living_a0", dict(memory_kv=True, memory_decay=True, rag_lambda=0.0, **MEM)),
    ("kv_static_a0", dict(memory_kv=True, memory_decay=False, rag_lambda=0.0, **MEM)),
    ("kv_living_a05", dict(memory_kv=True, memory_decay=True, rag_lambda=0.5, **MEM)),
    ("kv_living_a10", dict(memory_kv=True, memory_decay=True, rag_lambda=1.0, **MEM)),
]


def part_code(bundle, summary) -> None:
    files = [p.read_text(encoding="utf-8") for p in sorted(PROJECT.rglob("*.py"))]
    nodes = build_project_tree(PROJECT)
    ctx = project_context()
    print(f"\n########## ЧАСТЬ A: КОД ({len(nodes)} узлов) ##########", flush=True)
    for cond, use_bank, decay, alpha in CODE_CONDS:
        bank = SemTreeBank(nodes, space=SPACE, tau=TAU, device=bundle.device,
                           use_bank=use_bank, alpha=alpha, decay=decay)
        t0 = time.perf_counter()
        bank.ingest(bundle)
        bank.recluster(SPACE, TAU)
        ingest_s = time.perf_counter() - t0
        n_slots = bank.bank.count_element if bank.bank is not None else 0
        print(f"\n=== code/{cond} === ингестия {ingest_s:.0f}s, слотов {n_slots}", flush=True)
        entries = []
        for task in TASKS:
            res, diag = bank.retrieve(bundle, task["user"], k_clusters=K_CLUSTERS, k=K)
            notes = [n.text for n, _, _ in res]
            answer = generate(bundle, ctx, notes, task["user"])
            entry = score_entry(task, answer, notes)
            entry["def_hits"] = {n: any(f"def {n}" in note for note in notes)
                                 for n in task["expected"]}
            entries.append(entry)
            print(f"  {task['id']}: reuse={entry['reuse']} def={entry['def_hits']}", flush=True)
        reuse = sum(sum(e["reuse"].values()) for e in entries)
        hits = sum(sum(e["retrieval_hit"].values()) for e in entries)
        defs = sum(sum(e["def_hits"].values()) for e in entries)
        summary["code"][cond] = {"reuse": reuse, "hits": hits, "def_hits": defs,
                                 "ingest_s": round(ingest_s, 1), "slots": n_slots,
                                 "gist_used": bank.stats.get("gist_used", 0), "entries": entries}
        (DIR / f"code_{cond}.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def part_dialogue(bundle, summary) -> None:
    session = build_thread_dialogue()
    print(f"\n########## ЧАСТЬ B: ДИАЛОГ ({len(session.user_turns())} ходов) ##########", flush=True)
    for cond, mem in DLG_CONDS:
        strat = Strategy(cond, context="window", window=2, rag=True, rag_centered=True,
                         rag_layer="last", **mem)
        t0 = time.perf_counter()
        print(f"\n=== dlg/{cond} ===", flush=True)
        run = run_session(bundle, session, strat, max_new_tokens=48, seed=0)
        a = analyze_session({cond: run})[cond]
        summary["dialogue"][cond] = {
            "recall": a["recall_mean"], "confusion": a["stale_rate"],
            "tone": a.get("tone_mean"), "wall_s": round(time.perf_counter() - t0, 1),
            "kv_injected": run.kv_injected,
            "probes": [{"q": d["question"], "score": d["score"], "stale": d["stale_score"],
                        "answer": d["answer"]} for d in a["probe_details"]]}
        (DIR / f"dlg_{cond}.json").write_text(json.dumps(summary["dialogue"][cond], indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  recall={a['recall_mean']} confusion={a['stale_rate']} tone={a.get('tone_mean')}", flush=True)


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    summary = {"model": "qwen35-4b", "device": device, "code": {}, "dialogue": {}}
    part_code(bundle, summary)
    part_dialogue(bundle, summary)
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Живая память как ядро: TTL-динамика и gist-readout",
        "",
        "Тезис: ядро — живая TTL-память темы (refresh/decay/eviction), gist — один readout.",
        "decay on/off = живая vs статическая; alpha/lambda = сила gist (0 = без gist).",
        "",
        "## Часть A — код (taskq_large, semtree mean/tau=0.6)",
        "",
        "| Условие | Reuse /10 | Hits /10 | Def hits /10 | Слотов | gist_used |",
        "|---|---|---|---|---|---|",
    ]
    for cond, c in s["code"].items():
        lines.append(f"| {cond} | {c['reuse']} | {c['hits']} | {c['def_hits']} | {c['slots']} | {c['gist_used']} |")
    lines += ["", "## Часть B — диалог (thread-тест, дрейф темы)", "",
              "| Условие | recall | confusion (потеря нити) | tone | K/V inj |",
              "|---|---|---|---|---|"]
    for cond, c in s["dialogue"].items():
        tone = f"{c['tone']:.3f}" if c.get("tone") is not None else "—"
        lines.append(f"| {cond} | {c['recall']:.3f} | {c['confusion']:.3f} | {tone} | {c['kv_injected']} |")
    lines += ["",
              "Часть A: банк влияет на поиск через gist (alpha=0 = базлайн без банка);",
              "living vs static при alpha>0 изолирует TTL-динамику. Часть B: K/V-банк входит",
              "в генерацию напрямую, lambda=0 = живая память БЕЗ gist; kv_static_a0 = статический банк.",
              "Ниже confusion = лучше удержание нити. Данные: `code_<cond>.json`, `dlg_<cond>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

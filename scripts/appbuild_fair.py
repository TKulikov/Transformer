"""ЧЕСТНЫЙ app-building тест живой памяти (заказ 15.11): длинная сборка, без утечки.

Прежний appbuild был вырожден (запросы называли reuse-цели). Здесь — честно:
запросы называют НОВУЮ функцию, но НЕ называют prior-функции для переиспользования;
агент должен САМ вспомнить свой ранний код из памяти. window=1 форсирует опору
на память (поздний шаг не видит ранний код). Полная живая система (K/V с фиксом
prefill + centered RAG + gist + merge) против clean RAG и статических чанков.

Скоринг reuse-vs-rewrite через AST: should-reuse функция ВЫЗЫВАЕТСЯ (reuse) /
ПЕРЕОПРЕДЕЛЯЕТСЯ (rewrite) / отсутствует (miss). 3 сида для статистики.

Условия:
* clean_rag     — centered RAG, окно, без памяти своего кода;
* static_chunks — gen_chunks (статический поиск своего кода), без живой памяти;
* living_full   — gen_chunks + memory_kv (K/V) + rag_lambda (gist) + merge.

Выход: results/appbuild_fair/{summary.json,REPORT.md,<cond>_s<seed>.json}
Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/appbuild_fair.py
"""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from tmem.dialogue import Strategy, run_session  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.qa import DialogSession  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "appbuild_fair"
SEEDS = [0, 1, 2]
# (id новой функции, запрос без имён reuse-целей, should_reuse = свой prior-код)
STEPS = [
    ("validate_record", "Write a function validate_record(record) that checks a record has the required fields id and amount. Reply with code only.", []),
    ("normalize_amount", "Write a function normalize_amount(records) that normalizes the amount field of each record to a safe numeric range. Reply with code only.", []),
    ("dedupe_records", "Write a function dedupe_records(records) that removes duplicate records by id, keeping the first. Reply with code only.", []),
    ("process_batch", "Write a function process_batch(records) that validates each record, normalizes amounts, and removes duplicates. Reply with code only.", ["validate_record", "normalize_amount", "dedupe_records"]),
    ("retry_failed", "Write a function retry_failed(fn, attempts) that retries a failing operation up to attempts times. Reply with code only.", []),
    ("process_with_retry", "Write a function process_with_retry(records) that processes a batch of records, retrying on failure. Reply with code only.", ["process_batch", "retry_failed"]),
    ("summarize_results", "Write a function summarize_results(results) that aggregates processing results into counts and totals. Reply with code only.", []),
    ("run_pipeline", "Write a function run_pipeline(records) that processes records with retry and then summarizes the results. Reply with code only.", ["process_with_retry", "summarize_results"]),
    ("filter_active", "Write a function filter_active(records) that keeps only records marked active. Reply with code only.", []),
    ("full_pipeline", "Write a function full_pipeline(records) that filters active records and then runs the whole processing pipeline on them. Reply with code only.", ["filter_active", "run_pipeline"]),
]
CONDITIONS = {
    "clean_rag": dict(gen_chunks=False, memory=False, memory_kv=False, rag_lambda=0.0),
    "static_chunks": dict(gen_chunks=True, memory=False, memory_kv=False, rag_lambda=0.0,
                          chunk_embed="span", chunk_key="content", chunk_key_layers="mean", chunk_k=4),
    "living_full": dict(gen_chunks=True, memory_kv=True, rag_lambda=0.5,
                        memory_merge_k=0.25, memory_crit=0.95, memory_window_crit=0.5, kv_slots=32,
                        chunk_embed="span", chunk_key="content", chunk_key_layers="mean", chunk_k=4),
}


def score_reuse(answer: str, should_reuse: list[str]) -> dict:
    """AST: вызов (reuse) vs переопределение (rewrite) vs отсутствие (miss)."""
    try:
        tree = ast.parse(answer)
    except SyntaxError:
        # вырезаем код из markdown-заборов и пробуем снова
        import re
        m = re.search(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
        try:
            tree = ast.parse(m.group(1) if m else answer)
        except SyntaxError:
            return {n: "miss" for n in should_reuse}
    called, defined = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
    out = {}
    for n in should_reuse:
        if n in called and n not in defined:
            out[n] = "reuse"
        elif n in defined:
            out[n] = "rewrite"
        elif n in called:
            out[n] = "reuse"
        else:
            out[n] = "miss"
    return out


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    requests = [s[1] for s in STEPS]
    session = DialogSession(name="appbuild-fair", opening=requests, fillers=[], probes=[])
    summary = {"model": "qwen35-4b", "device": device, "window": 1, "seeds": SEEDS,
               "n_steps": len(STEPS), "conditions": {}}

    for cond, cfg in CONDITIONS.items():
        per_seed = []
        for seed in SEEDS:
            strat = Strategy(f"{cond}_s{seed}", context="window", window=1, rag=True,
                             rag_centered=True, rag_layer="last", **cfg)
            t0 = time.perf_counter()
            run = run_session(bundle, session, strat, max_new_tokens=256, seed=seed)
            answers = [t.answer_text for t in run.turns]
            steps_res = []
            reuse_n = rewrite_n = miss_n = total = 0
            for (sid, _req, should), ans in zip(STEPS, answers):
                sc = score_reuse(ans, should)
                steps_res.append({"step": sid, "should_reuse": should, "score": sc, "answer": ans})
                for v in sc.values():
                    total += 1
                    reuse_n += v == "reuse"; rewrite_n += v == "rewrite"; miss_n += v == "miss"
            entry = {"seed": seed, "reuse": reuse_n, "rewrite": rewrite_n, "miss": miss_n,
                     "total": total, "wall_s": round(time.perf_counter() - t0, 1),
                     "kv_injected": run.kv_injected, "steps": steps_res}
            per_seed.append(entry)
            (DIR / f"{cond}_s{seed}.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  {cond} seed{seed}: reuse={reuse_n}/{total} rewrite={rewrite_n} miss={miss_n} "
                  f"kv={run.kv_injected} ({entry['wall_s']}s)", flush=True)
        summary["conditions"][cond] = {
            "reuse_mean": round(sum(e["reuse"] for e in per_seed) / len(SEEDS), 2),
            "rewrite_mean": round(sum(e["rewrite"] for e in per_seed) / len(SEEDS), 2),
            "miss_mean": round(sum(e["miss"] for e in per_seed) / len(SEEDS), 2),
            "total": per_seed[0]["total"], "per_seed": per_seed}
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = ["# ЧЕСТНЫЙ app-building: живая память помнит СВОЙ код (без утечки, AST-скоринг)", "",
             f"{s['n_steps']} шагов, window={s['window']}, сиды {s['seeds']}.",
             "reuse = вызов prior-функции; rewrite = переопределение; miss = отсутствие.", "",
             "| Условие | reuse (mean) | rewrite | miss | всего |", "|---|---|---|---|---|"]
    for cond, c in s["conditions"].items():
        lines.append(f"| {cond} | {c['reuse_mean']} | {c['rewrite_mean']} | {c['miss_mean']} | {c['total']} |")
    lines += ["", "## По сидам (reuse)", "",
              "| Условие | " + " | ".join(f"seed{sd}" for sd in s["seeds"]) + " |",
              "|---|" + "---|" * len(s["seeds"])]
    for cond, c in s["conditions"].items():
        cells = [str(e["reuse"]) for e in c["per_seed"]]
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["", "Гипотеза: living_full > static_chunks > clean_rag по reuse своего кода.",
              "clean_rag — окно без памяти кода; static_chunks — статический поиск; living_full — вся живая система.",
              "Данные: `<cond>_s<seed>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

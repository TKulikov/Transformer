"""Бенчмарк стоимости поиска: латентность и ресурсы по режимам.

Заказ 15.11: паритет ли по latency/ресурсам между flat/descent/union/shallow?

Меряем на большом проекте taskq_large:
* ингестия банка (общая для всех режимов, mode-independent) — wall + число
  embed-форвардов;
* латентность retrieve по режимам (мс): эмбеддинг запроса общий для всех,
  поэтому РАЗНИЦА между режимами изолирует стоимость скоринга;
* размер injected notes (число + токены) — прокси стоимости prefill генерации;
* оценка числа cosine-операций скоринга на запрос.

Сама генерация (192 токена decode) одинакова для всех режимов; отличается
только prefill ∝ размеру notes. Ингестия — основная upfront-стоимость.

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/retrieval_cost_bench.py
Выход:  results/chunk_code/retrieval_cost/{summary.json,REPORT.md}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from tmem.hiermem import HierLatentBank, build_project_tree  # noqa: E402
from tmem.models import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "testbeds" / "taskq_large" / "taskq_large"
DIR = ROOT / "results" / "chunk_code" / "retrieval_cost"
MODES = ["flat", "descent", "union", "shallow"]
QUERIES = [
    "reprocess failed records with exponential backoff using existing helpers",
    "parse a timeout string into milliseconds and clamp it to a safe range",
    "deduplicate record keys and split them into fixed-size batches",
    "slugify a record name and compute the total of a numeric field",
    "return an aggregate summary of invoice records using the existing service",
]
REPEATS = 12


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    nodes = build_project_tree(PROJECT)
    n_decl = sum(1 for n in nodes if n.kind == "function")
    bank = HierLatentBank(nodes, beta=1.0, device=bundle.device)
    t0 = time.perf_counter()
    bank.ingest(bundle)
    if device == "cuda":
        torch.cuda.synchronize()
    ingest_s = time.perf_counter() - t0
    # embed-форварды: функция = 2 (key+value), прочее = 1
    n_fwd = sum(2 if n.kind == "function" else 1 for n in nodes)
    print(f"ингестия: {ingest_s:.1f}s, узлов {len(nodes)}, объявлений {n_decl}, "
          f"embed-форвардов ~{n_fwd}", flush=True)

    tok = bundle.tokenizer
    summary = {"device": device, "nodes": len(nodes), "declarations": n_decl,
               "ingest_s": round(ingest_s, 2), "embed_forwards": n_fwd,
               "repeats": REPEATS, "modes": {}}
    for mode in MODES:
        # прогрев
        bank.retrieve(bundle, QUERIES[0], k_branches=5, k_children=2, mode=mode)
        times, note_counts, note_tokens = [], [], []
        for _ in range(REPEATS):
            for q in QUERIES:
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                res = bank.retrieve(bundle, q, k_branches=5, k_children=2, mode=mode)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
                notes = [n.text for n, _, _ in res]
                note_counts.append(len(notes))
                note_tokens.append(sum(len(tok(n)["input_ids"]) for n in notes))
        times.sort()
        mean_ms = sum(times) / len(times)
        med_ms = times[len(times) // 2]
        summary["modes"][mode] = {
            "retrieve_mean_ms": round(mean_ms, 2),
            "retrieve_median_ms": round(med_ms, 2),
            "retrieve_p90_ms": round(times[int(len(times) * 0.9)], 2),
            "mean_notes": round(sum(note_counts) / len(note_counts), 2),
            "mean_note_tokens": round(sum(note_tokens) / len(note_tokens), 1),
        }
        print(f"{mode:9s} retrieve {mean_ms:6.2f} ms (med {med_ms:.2f}), "
              f"notes {summary['modes'][mode]['mean_notes']}, "
              f"note_tokens {summary['modes'][mode]['mean_note_tokens']}", flush=True)
    (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary)
    print("готово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Стоимость поиска по режимам (taskq_large)",
        "",
        f"Узлов: {s['nodes']}, объявлений: {s['declarations']}. "
        f"Ингестия (общая для всех режимов): {s['ingest_s']} с, ~{s['embed_forwards']} embed-форвардов.",
        f"Латентность retrieve: {s['repeats']} повторов × 5 запросов, мс. "
        "Эмбеддинг запроса общий → разница между режимами = стоимость скоринга.",
        "",
        "| Режим | retrieve mean, мс | median | p90 | notes (шт) | note tokens (prefill) |",
        "|---|---|---|---|---|---|",
    ]
    for mode, m in s["modes"].items():
        lines.append(f"| {mode} | {m['retrieve_mean_ms']} | {m['retrieve_median_ms']} "
                     f"| {m['retrieve_p90_ms']} | {m['mean_notes']} | {m['mean_note_tokens']} |")
    lines += [
        "",
        "Генерация (192 токена decode) одинакова для всех режимов; отличается только prefill ∝ note tokens.",
        "Ингестия — основная upfront-стоимость и не зависит от режима поиска.",
        "",
    ]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

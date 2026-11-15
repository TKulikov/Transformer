"""Оценка памяти на большом мультифайловом проекте taskq_large (~28k, 82 файла).

Заказ 15.11: проверить shallow (минимальное дерево файл→класс + плоский поиск
внутри) против flat/descent/union на МАСШТАбе + множестве файлов, где
диверсификация по файлам должна проявиться (на маленьком проекте flat и так
справлялся, а мегафайл — один файл, группировка вырождена).

Пробы требуют вспомнить хелперы из core.utils и методы сервисов из разных
ресурсных файлов. Контекст — описание проекта без тел хелперов (нет утечки).
Метрики: reuse, retrieval hit (def ИЛИ вызов), валидность.

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/taskq_large_eval.py
Выход:  results/chunk_code/taskq_large/{summary.json,REPORT.md,<mode>.json}
"""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from tmem.generate import decode_causal  # noqa: E402
from tmem.hiermem import HierLatentBank, build_project_tree  # noqa: E402
from tmem.kvcache import KVConfig  # noqa: E402
from tmem.models import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "testbeds" / "taskq_large" / "taskq_large"
DIR = ROOT / "results" / "chunk_code" / "taskq_large"

TASKS = [
    {"id": "retry_backoff", "expected": ["retry", "backoff_delay"],
     "user": "Write a function reprocess_failed(records) that reprocesses failed records with "
             "exponential backoff, reusing the project's existing helpers. Reply with code only."},
    {"id": "parse_clamp", "expected": ["parse_duration", "clamp"],
     "user": "Write a function safe_timeout(text) that parses a timeout string into milliseconds "
             "and clamps it to a safe range, reusing existing helpers. Reply with code only."},
    {"id": "batch_dedupe", "expected": ["dedupe", "chunked"],
     "user": "Write a function prepare_batches(keys, size) that deduplicates record keys and "
             "splits them into fixed-size batches, reusing existing helpers. Reply with code only."},
    {"id": "slug_total", "expected": ["slugify", "compute_total"],
     "user": "Write a function name_and_total(records, field) that slugifies a record name and "
             "computes the total of a numeric field over records, reusing existing helpers. "
             "Reply with code only."},
    {"id": "invoice_summary", "expected": ["summarize", "domain_metric"],
     "user": "Write a handler method that returns an aggregate summary of invoice records using "
             "the existing invoice service. Reply with code only."},
]
MODES = ["no_memory", "flat", "descent", "union", "shallow"]


def project_context() -> str:
    return ("Project taskq_large: a layered client-server resource API. Layers: core (utils, "
            "protocol, errors), models, repositories, services, handlers; one class per file. "
            "Shared helpers live in core.utils; each resource has a Service with a summarize "
            "method and a domain_metric.")


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    nodes = build_project_tree(PROJECT)
    bank = HierLatentBank(nodes, beta=1.0, device=bundle.device)
    t0 = time.perf_counter()
    bank.ingest(bundle)
    ingest_s = time.perf_counter() - t0
    kinds = {k: sum(1 for n in nodes if n.kind == k) for k in ("file", "class", "function", "usage")}
    print(f"дерево: {kinds}, ингестия {ingest_s:.0f}s", flush=True)
    ctx = project_context()

    summary = {"model": "qwen35-4b", "device": device, "project": "taskq_large",
               "tree": kinds, "ingest_s": round(ingest_s, 1), "modes": {}}
    for mode in MODES:
        print(f"\n=== {mode} ===", flush=True)
        entries = []
        for task in TASKS:
            notes = []
            if mode != "no_memory":
                res = bank.retrieve(bundle, task["user"], k_branches=5, k_children=2, mode=mode)
                notes = [n.text for n, _, _ in res]
            nb = ("Relevant notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n") if notes else ""
            prompt = bundle.to_chat(f"Dialogue.\n{nb}{ctx}\nUser: {task['user']}\nAssistant:")
            ids = bundle.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)["input_ids"]
            ids = ids[:, -bundle.max_prompt_len(reserve=192):].to(bundle.device)
            dec = decode_causal(bundle.model, ids, kv=KVConfig.classic(), cache="use",
                                max_new_tokens=192, do_sample=False, eos_ids=bundle.eos_ids(), seed=0)
            answer = bundle.tokenizer.decode(dec.token_ids, skip_special_tokens=True).strip()
            try:
                ast.parse(answer if "def " in answer else "pass")
                valid = True
            except SyntaxError:
                valid = False
            entry = {"task": task["id"], "valid_code": valid,
                     "reuse": {n: (n in answer) for n in task["expected"]},
                     "retrieval_hit": {n: any((f"def {n}" in note) or (f"{n}(" in note)
                                              for note in notes) for n in task["expected"]},
                     "answer": answer, "injected": notes}
            entries.append(entry)
            print(f"  {task['id']}: reuse={entry['reuse']} hits={entry['retrieval_hit']}", flush=True)
        summary["modes"][mode] = entries
        (DIR / f"{mode}.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Память на большом мультифайловом проекте taskq_large",
        "",
        f"Дерево: {s['tree']}; ингестия {s['ingest_s']} с. Контекст — описание проекта без тел хелперов.",
        "Retrieval hit — имя в notes как def ИЛИ вызов. Макс reuse/hits = 10 (5 проб × 2).",
        "",
        "| Режим | Задача | Reuse | Retrieval hit |",
        "|---|---|---|---|",
    ]
    for mode, entries in s["modes"].items():
        for e in entries:
            reuse = ", ".join(f"{k}={'да' if v else 'нет'}" for k, v in e["reuse"].items())
            hit = ", ".join(f"{k}={'да' if v else 'нет'}" for k, v in e["retrieval_hit"].items())
            lines.append(f"| {mode} | {e['task']} | {reuse} | {hit} |")
    lines += ["", "## Сводка (сумма по пробам, макс 10)", ""]
    for mode, entries in s["modes"].items():
        total = sum(sum(e["reuse"].values()) for e in entries)
        hits = sum(sum(e["retrieval_hit"].values()) for e in entries)
        lines.append(f"- {mode}: reuse {total}/10, retrieval hits {hits}/10")
    lines += ["", "Данные: `summary.json`, `<mode>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

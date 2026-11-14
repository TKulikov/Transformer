"""Оценка иерархической памяти (дерево файл→класс→метод→вызовы) на 100k файле.

Сравнение на тех же пробах «дописать», что в chunk_stress_100k, но с ЧИСТЫМ
хвостом: 40-строчное окно без имён хелперов (в стресс-тесте хвост leaking
имена, и no_memory решал пробы без памяти). Условия: no_memory, плоский
контентный банк (flat_ck), иерархический (hier: объявления по абсолютным
ключам + спуск к использованиям по отклонениям от родителя).

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/chunk_hier_eval.py
Выход:  results/chunk_code/hier100k/{summary.json,REPORT.md,<cond>.json}
"""

from __future__ import annotations

import ast
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from tmem.chunkmem import ChunkBank, chunk_text, embed_block_key, embed_chunk  # noqa: E402
from tmem.generate import decode_causal  # noqa: E402
from tmem.hiermem import HierBank, build_code_tree  # noqa: E402
from tmem.kvcache import KVConfig  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.semantic import embed_utterance  # noqa: E402

DIR = Path(__file__).resolve().parent.parent / "results" / "chunk_code" / "hier100k"
SRC = Path(__file__).resolve().parent.parent / "results" / "chunk_code" / "stress100k" / "bigcode.py"
HELPERS = ["normalize_records", "filter_active", "compute_total", "group_by",
           "moving_average", "clamp", "slugify", "parse_int", "format_table", "dedupe"]

TASKS = [
    {"id": "continue", "expected": [],
     "user": "Continue this Python file with the next function in the same style. Reply with code only."},
    {"id": "named_early", "expected": ["normalize_records", "compute_total"],
     "user": "Write build_summary_report(records) that uses normalize_records and compute_total "
             "over the 'amount' field. Reply with code only."},
    {"id": "clean_mid", "expected": ["group_by", "moving_average"],
     "user": "Write trend_report(values_by_group) that groups records by region and smooths each "
             "group's numeric series before averaging. Reuse the existing helpers from this file. "
             "Reply with code only."},
    {"id": "clean_total", "expected": ["filter_active", "compute_total"],
     "user": "Write export_active_total(records) that keeps only active records and returns their "
             "total 'amount'. Reuse the existing helpers from this file. Reply with code only."},
]


def clean_tail(text: str, n_lines: int = 40) -> tuple[str, dict]:
    """Хвост без имён хелперов, чтобы контекст не leaking ответы.

    Сначала ищем по всему файлу окно без единого упоминания хелпера.
    Если нет — берём хвост и ВЫБРАСЫВАЕМ строки с хелперами (без placeholder:
    маскировка токеном вроде _redacted_ загрязняет метрику — модель копирует
    его в ответ как имя функции).
    """
    lines = text.splitlines()
    for start in range(len(lines) - n_lines, -1, -1):
        window = lines[start:start + n_lines]
        joined = "\n".join(window)
        if not any(h in joined for h in HELPERS):
            return joined, {"line_start": start, "helper_mentions": 0}
    window = lines[-n_lines:]
    kept = [ln for ln in window if not any(h in ln for h in HELPERS)]
    return "\n".join(kept), {"line_start": len(lines) - n_lines,
                             "dropped_lines": n_lines - len(kept)}


def extract_code(answer: str) -> str:
    m = re.search(r"```python\s*(.*?)```", answer, re.S)
    return m.group(1) if m else answer


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    text = SRC.read_text(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)

    tail, tail_meta = clean_tail(text)
    print("хвост:", json.dumps(tail_meta), flush=True)

    nodes = build_code_tree(text)
    hier = HierBank(nodes, device=bundle.device)
    t0 = time.perf_counter()
    hier.ingest(bundle)
    hier_s = time.perf_counter() - t0
    print(f"hier ингестия: {hier.stats} за {hier_s:.0f}s", flush=True)

    chunks = chunk_text(text, max_tokens=256, tokenizer=bundle.tokenizer)
    flat = ChunkBank(bundle.hidden_size, capacity=8192, centered=True, device=bundle.device)
    t0 = time.perf_counter()
    for i, ch in enumerate(chunks):
        context = "\n".join(chunks[max(0, i - 2):i])
        key, _ = embed_block_key(bundle, ch, layers="mean", max_tokens=256)
        val, _, _ = embed_chunk(bundle, ch, context=context, mode="span",
                                max_tokens=256, context_tokens=512, layer="mid")
        flat.add(ch, key_emb=key, val_emb=val, turn=i)
    flat_s = time.perf_counter() - t0
    print(f"flat ингестия: {len(flat.texts)} чанков за {flat_s:.0f}s", flush=True)

    summary = {"model": "qwen35-4b", "device": device, "tail": tail_meta,
               "value_layer": "mid (латентный слой смыслов)",
               "tree": {"nodes": len(nodes),
                        "kinds": {k: sum(1 for n in nodes if n.kind == k)
                                  for k in ("file", "class", "function", "usage")}},
               "ingest": {"hier_wall_s": round(hier_s, 1), "flat_wall_s": round(flat_s, 1),
                          "hier_stats": hier.stats},
               "conditions": {}}
    for cond in ("no_memory", "flat_ck", "hier"):
        print(f"\n=== {cond} ===", flush=True)
        entries = []
        for task in TASKS:
            notes, branches = [], set()
            if cond == "flat_ck":
                q = embed_utterance(bundle, task["user"])
                notes = [t for t, _, _ in flat.retrieve(q, k=4, score="key")]
            elif cond == "hier":
                q = embed_utterance(bundle, task["user"])
                for node, score, stage in hier.retrieve(q, k_branches=3, k_children=2):
                    notes.append(node.text)
                    if stage == "declaration":
                        branches.add(node.name)
            note_block = ("Relevant notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n") if notes else ""
            prompt = bundle.to_chat(f"Dialogue.\n{note_block}{tail}\nUser: {task['user']}\nAssistant:")
            ids = bundle.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)["input_ids"]
            ids = ids[:, -bundle.max_prompt_len(reserve=192):].to(bundle.device)
            dec = decode_causal(bundle.model, ids, kv=KVConfig.classic(), cache="use",
                                max_new_tokens=192, do_sample=False,
                                eos_ids=bundle.eos_ids(), seed=0)
            answer = bundle.tokenizer.decode(dec.token_ids, skip_special_tokens=True).strip()
            try:
                ast.parse(extract_code(answer))
                valid = True
            except SyntaxError:
                valid = False
            entry = {"task": task["id"], "valid_code": valid,
                     "reuse": {n: (n in answer) for n in task["expected"]},
                     "retrieval_hit": {n: any(f"def {n}" in note for note in notes)
                                       for n in task["expected"]},
                     "branches": sorted(branches), "n_notes": len(notes),
                     "answer": answer, "injected": notes}
            entries.append(entry)
            print(f"  {task['id']}: valid={valid} reuse={entry['reuse']} "
                  f"hits={entry['retrieval_hit']} branches={sorted(branches)}", flush=True)
        summary["conditions"][cond] = entries
        (DIR / f"{cond}.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Иерархическая память на файле ~100k токенов",
        "",
        "Дерево файл → класс → метод → вызовы; ключ узла = среднее активаций блока, "
        "центрирование отклонением от родителя (метод от класса, вызов от объявления); "
        "поиск: объявления по абсолютным ключам → спуск к использованиям по residual-ключам.",
        "",
        f"- Дерево: {s['tree']['nodes']} узлов {s['tree']['kinds']}.",
        f"- Ингестия: hier {s['ingest']['hier_wall_s']} с, flat {s['ingest']['flat_wall_s']} с.",
        f"- Хвост промпта: {s['tail']} (имена хелперов не leaking).",
        "",
        "| Условие | Задача | Валид | Reuse | Retrieval hit | Ветки |",
        "|---|---|---|---|---|---|",
    ]
    for cond, entries in s["conditions"].items():
        for e in entries:
            reuse = ", ".join(f"{k}={'да' if v else 'нет'}" for k, v in e["reuse"].items()) or "—"
            hit = ", ".join(f"{k}={'да' if v else 'нет'}" for k, v in e["retrieval_hit"].items()) or "—"
            lines.append(f"| {cond} | {e['task']} | {'да' if e['valid_code'] else 'нет'} "
                         f"| {reuse} | {hit} | {len(e['branches']) or '—'} |")
    lines += ["", "Данные: `summary.json`, `<condition>.json` (ответы, injected notes).", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

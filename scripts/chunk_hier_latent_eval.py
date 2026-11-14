"""Гибридная память для кода: дерево + латентный слой + TTL-gist, на 100k.

Целевой вариант (заказ 14.11): структура дерева как скелет хранения +
латентные эмбеддинги узлов (ключ объявления = norm(q_сигнатура + beta·тело)
на одном латентном слое, residual-центрирование от родителя) + запрос,
контекстуемый TTL-gist слот-банка. Те же пробы «дописать» и чищеный хвост,
что в трёхстороннем тесте, для сопоставимости.

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/chunk_hier_latent_eval.py
Очередь: scripts/queue_hier_latent.sh (ждёт окончания chunk_three_way).
Выход:  results/chunk_code/hierlatent100k/{summary.json,REPORT.md}
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

from chunk_hier_eval import SRC, TASKS, clean_tail, extract_code  # noqa: E402
from tmem.generate import decode_causal  # noqa: E402
from tmem.hiermem import HierLatentBank, build_code_tree  # noqa: E402
from tmem.kvcache import KVConfig  # noqa: E402
from tmem.models import load_model  # noqa: E402

DIR = Path(__file__).resolve().parent.parent / "results" / "chunk_code" / "hierlatent100k"
THREE = Path(__file__).resolve().parent.parent / "results" / "chunk_code" / "threeway100k" / "summary.json"
BETA, ALPHA = 1.0, 0.5


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    text = SRC.read_text(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    tail, tail_meta = clean_tail(text)
    nodes = build_code_tree(text)
    bank = HierLatentBank(nodes, beta=BETA, device=bundle.device)
    t0 = time.perf_counter()
    bank.ingest(bundle)
    ingest_s = time.perf_counter() - t0
    print(f"ингестия hybrids: {bank.stats} за {ingest_s:.0f}s", flush=True)

    entries = []
    for task in TASKS:
        res = bank.retrieve(bundle, task["user"], k_branches=3, k_children=2, alpha=ALPHA)
        notes = [n.text for n, _, _ in res]
        branches = sorted({n.name for n, _, st in res if st == "declaration"})
        nb = ("Relevant notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n") if notes else ""
        prompt = bundle.to_chat(f"Dialogue.\n{nb}{tail}\nUser: {task['user']}\nAssistant:")
        ids = bundle.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)["input_ids"]
        ids = ids[:, -bundle.max_prompt_len(reserve=192):].to(bundle.device)
        dec = decode_causal(bundle.model, ids, kv=KVConfig.classic(), cache="use",
                            max_new_tokens=192, do_sample=False, eos_ids=bundle.eos_ids(), seed=0)
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
                 "branches": branches, "answer": answer, "injected": notes}
        entries.append(entry)
        print(f"  {task['id']}: valid={valid} reuse={entry['reuse']} "
              f"hits={entry['retrieval_hit']} branches={branches}", flush=True)
    summary = {"model": "qwen35-4b", "device": device, "tail": tail_meta,
               "beta": BETA, "alpha": ALPHA, "variant": "hier+latent+ttl-gist",
               "ingest": {"wall_s": round(ingest_s, 1), "stats": bank.stats},
               "entries": entries}
    (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Гибридная память для кода: дерево + латентный слой + TTL-gist",
        "",
        f"beta={s['beta']}, alpha={s['alpha']}; ингестия {s['ingest']}.",
        "",
        "| Задача | Валид | Reuse | Retrieval hit | Ветки |",
        "|---|---|---|---|---|",
    ]
    for e in s["entries"]:
        reuse = ", ".join(f"{k}={'да' if v else 'нет'}" for k, v in e["reuse"].items()) or "—"
        hit = ", ".join(f"{k}={'да' if v else 'нет'}" for k, v in e["retrieval_hit"].items()) or "—"
        lines.append(f"| {e['task']} | {'да' if e['valid_code'] else 'нет'} | {reuse} | {hit} "
                     f"| {len(e['branches'])} |")
    if THREE.exists():
        three = json.loads(THREE.read_text(encoding="utf-8"))
        lines += ["", "## Сравнение с тремя вариантами (reuse на чистых пробах)", "",
                  "| Вариант | clean_mid | clean_total |", "|---|---|---|"]
        for cond, ents in three["conditions"].items():
            cm = next((e for e in ents if e["task"] == "clean_mid"), {})
            ct = next((e for e in ents if e["task"] == "clean_total"), {})
            lines.append(f"| {cond} | {sum(cm.get('reuse', {}).values())}/2 "
                         f"| {sum(ct.get('reuse', {}).values())}/2 |")
        hl = {e["task"]: e for e in s["entries"]}
        lines.append(f"| **hier+latent** | {sum(hl['clean_mid']['reuse'].values())}/2 "
                     f"| {sum(hl['clean_total']['reuse'].values())}/2 |")
    lines += ["", "Данные: `summary.json` (ответы, injected notes).", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

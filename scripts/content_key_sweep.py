"""Свип content-вектора (вариант 3): key = norm(q_rec + β·latent + γ·content).

Заказ 15.11: content = изолированный mid-mean тела функции (содержание по
принципу энкодера), latent = контекстный вектор активаций. Проверяем, даёт
 ли добавление content-вектора выигрыш в поиске. Одна ингестия (компоненты
сохраняются на узлах), затем rekey(β=1, γ) для каждого γ — без повторных
embed-форвардов. Поиск flat (рабочая лошадка).

γ=0 — базовый ключ (без content); γ>0 — вклад содержания.

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/content_key_sweep.py
Выход:  results/chunk_code/content_key_sweep/{summary.json,REPORT.md}
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

from taskq_large_eval import PROJECT, TASKS, project_context  # noqa: E402
from tmem.generate import decode_causal  # noqa: E402
from tmem.hiermem import HierLatentBank, build_project_tree  # noqa: E402
from tmem.kvcache import KVConfig  # noqa: E402
from tmem.models import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "chunk_code" / "content_key_sweep"
GAMMAS = [0.0, 0.5, 1.0, 2.0]
BETA = 1.0


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    nodes = build_project_tree(PROJECT)
    bank = HierLatentBank(nodes, beta=BETA, gamma=1.0, device=bundle.device)
    t0 = time.perf_counter()
    bank.ingest(bundle)
    ingest_s = time.perf_counter() - t0
    print(f"ингестия (3 форварда/функцию): {ingest_s:.0f}s, узлов {len(nodes)}", flush=True)
    ctx = project_context()

    summary = {"model": "qwen35-4b", "device": device, "beta": BETA,
               "ingest_s": round(ingest_s, 1), "gammas": {}}
    for gamma in GAMMAS:
        bank.rekey(BETA, gamma)
        reuse_sum = hit_sum = 0
        per_task = []
        for task in TASKS:
            res = bank.retrieve(bundle, task["user"], k_branches=5, k_children=2, mode="flat")
            notes = [n.text for n, _, _ in res]
            nb = ("Relevant notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n") if notes else ""
            prompt = bundle.to_chat(f"Dialogue.\n{nb}{ctx}\nUser: {task['user']}\nAssistant:")
            ids = bundle.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)["input_ids"]
            ids = ids[:, -bundle.max_prompt_len(reserve=192):].to(bundle.device)
            dec = decode_causal(bundle.model, ids, kv=KVConfig.classic(), cache="use",
                                max_new_tokens=192, do_sample=False, eos_ids=bundle.eos_ids(), seed=0)
            answer = bundle.tokenizer.decode(dec.token_ids, skip_special_tokens=True).strip()
            reuse = {n: (n in answer) for n in task["expected"]}
            hit = {n: any((f"def {n}" in note) or (f"{n}(" in note) for note in notes)
                   for n in task["expected"]}
            reuse_sum += sum(reuse.values())
            hit_sum += sum(hit.values())
            per_task.append({"task": task["id"], "reuse": reuse, "hit": hit})
        summary["gammas"][str(gamma)] = {"reuse": reuse_sum, "hits": hit_sum, "tasks": per_task}
        print(f"γ={gamma}: reuse {reuse_sum}/10, hits {hit_sum}/10", flush=True)
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary)
    print("готово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Свип content-вектора (вариант 3), key = norm(q_rec + β·latent + γ·content)",
        "",
        f"β={s['beta']}; ингестия {s['ingest_s']} с (3 форварда/функцию). Поиск flat, проект taskq_large.",
        "content = изолированный mid-mean тела; latent = контекстный вектор активаций. Макс 10.",
        "",
        "| γ (content) | Reuse | Retrieval hits |",
        "|---|---|---|",
    ]
    for gamma, g in s["gammas"].items():
        lines.append(f"| {gamma} | {g['reuse']}/10 | {g['hits']}/10 |")
    lines += ["", "γ=0 — базовый ключ без content-вектора. Данные: `summary.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

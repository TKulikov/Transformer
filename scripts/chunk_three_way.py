"""Три варианта памяти на файле ~100k токенов: старый / дерево / латентный.

Заказ 14.11: сравнить на одних пробах «дописать» (чищеный хвост):
* old_flat — плоский контентный банк (ключ=mean всех слоёв, значение=mid,
  запрос=final) — как было;
* hier — дерево файл→класс→метод→вызовы, residual-центрирование от родителя;
* latent — LatentRagBank: запись = norm(q_rec + beta·answer) на латентном
  слое, запрос = norm(q + alpha·TTL-gist), всё в одном латентном пространстве.

Для latent «запрос» функции = сигнатура + docstring (прокси задачи),
«ответ» = тело функции; ингестия после «ответа», как в диалоге.

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/chunk_three_way.py
Выход:  results/chunk_code/threeway100k/{summary.json,REPORT.md,<cond>.json}
"""

from __future__ import annotations

import ast
import json
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from chunk_hier_eval import HELPERS, SRC, TASKS, clean_tail, extract_code  # noqa: E402
from tmem.chunkmem import ChunkBank, chunk_text, embed_block_key, embed_chunk  # noqa: E402
from tmem.generate import decode_causal  # noqa: E402
from tmem.hiermem import HierBank, build_code_tree  # noqa: E402
from tmem.kvcache import KVConfig  # noqa: E402
from tmem.latentrag import LatentRagBank  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.semantic import embed_utterance  # noqa: E402

DIR = Path(__file__).resolve().parent.parent / "results" / "chunk_code" / "threeway100k"
BETA, ALPHA = 1.0, 0.5


def request_proxy(node) -> str:
    src = textwrap.dedent(node.text)
    try:
        doc = ast.get_docstring(ast.parse(src))
    except SyntaxError:
        doc = None
    first = node.text.strip().splitlines()[0]
    return first + (" — " + doc if doc else "")


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    text = SRC.read_text(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    tail, tail_meta = clean_tail(text)
    nodes = build_code_tree(text)
    funcs = [n for n in nodes if n.kind == "function"]

    # --- old flat ---
    chunks = chunk_text(text, max_tokens=256, tokenizer=bundle.tokenizer)
    flat = ChunkBank(bundle.hidden_size, capacity=8192, centered=True, device=bundle.device)
    t0 = time.perf_counter()
    for i, ch in enumerate(chunks):
        ctx = "\n".join(chunks[max(0, i - 2):i])
        key, _ = embed_block_key(bundle, ch, layers="mean", max_tokens=256)
        val, _, _ = embed_chunk(bundle, ch, context=ctx, mode="span", max_tokens=256,
                                context_tokens=512, layer="mid")
        flat.add(ch, key_emb=key, val_emb=val, turn=i)
    flat_s = time.perf_counter() - t0

    # --- hier ---
    hier = HierBank(nodes, device=bundle.device)
    t0 = time.perf_counter()
    hier.ingest(bundle)
    hier_s = time.perf_counter() - t0

    # --- latent ---
    latent = LatentRagBank(bundle.hidden_size, beta=BETA, capacity=8192, device=bundle.device)
    t0 = time.perf_counter()
    for n in funcs:
        parent = nodes[n.parent] if n.parent is not None else None
        latent.add(bundle, request_proxy(n), n.text,
                   context=parent.text if parent else "", turn=n.nid)
    latent_s = time.perf_counter() - t0
    print(f"ингестия: flat {flat_s:.0f}s, hier {hier_s:.0f}s, latent {latent_s:.0f}s "
          f"({latent.stats})", flush=True)

    summary = {"model": "qwen35-4b", "device": device, "tail": tail_meta,
               "beta": BETA, "alpha": ALPHA,
               "ingest": {"flat_s": round(flat_s, 1), "hier_s": round(hier_s, 1),
                          "latent_s": round(latent_s, 1), "latent_stats": latent.stats},
               "conditions": {}}
    for cond in ("old_flat", "hier", "latent"):
        print(f"\n=== {cond} ===", flush=True)
        entries = []
        for task in TASKS:
            notes, branches = [], set()
            if cond == "old_flat":
                notes = [t for t, _, _ in flat.retrieve(embed_utterance(bundle, task["user"]),
                                                        k=4, score="key")]
            elif cond == "hier":
                for node, s, stage in hier.retrieve(embed_utterance(bundle, task["user"]),
                                                    k_branches=3, k_children=2):
                    notes.append(node.text)
                    if stage == "declaration":
                        branches.add(node.name)
            else:
                notes = [t for t, _, _ in latent.retrieve(bundle, task["user"], k=4, alpha=ALPHA)]
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
                     "branches": sorted(branches), "answer": answer, "injected": notes}
            entries.append(entry)
            print(f"  {task['id']}: valid={valid} reuse={entry['reuse']} "
                  f"hits={entry['retrieval_hit']}", flush=True)
        summary["conditions"][cond] = entries
        (DIR / f"{cond}.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Три варианта памяти на файле ~100k токенов",
        "",
        f"beta={s['beta']}, alpha={s['alpha']}; хвост: {s['tail']}.",
        f"Ингестия: {s['ingest']}.",
        "",
        "| Условие | Задача | Валид | Reuse | Retrieval hit |",
        "|---|---|---|---|---|",
    ]
    for cond, entries in s["conditions"].items():
        for e in entries:
            reuse = ", ".join(f"{k}={'да' if v else 'нет'}" for k, v in e["reuse"].items()) or "—"
            hit = ", ".join(f"{k}={'да' if v else 'нет'}" for k, v in e["retrieval_hit"].items()) or "—"
            lines.append(f"| {cond} | {e['task']} | {'да' if e['valid_code'] else 'нет'} | {reuse} | {hit} |")
    lines += ["", "Данные: `summary.json`, `<condition>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

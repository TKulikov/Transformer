"""Ретест поисков по коду на taskq_large (заказ 15.11, повтор).

Вчера на диалогах выяснилось, что латентные эмбеддинги (mid-слой) сильнее
hidden[-1], а латентный K/V-инжект преимущества не даёт. Здесь — ПОВТОРНАЯ
проверка на коде (мультифайловый проект taskq_large, 82 файла, ~28k токенов):
нужен ли латентный инжект для кода и какое хранение лучше — латентный банк
или hidden state[-1].

Условия (одни и те же 5 задач «дописать с reuse хелперов», greedy 192 токена):
* no_memory     — без памяти (базлайн);
* flat_last     — ChunkBank: ключ/значение/запрос на hidden[-1] (last);
* flat_mid      — ChunkBank: всё на латентном mid-слое (ретест «mid сильнее»);
* flat_legacy   — рецепт old_flat из threeway: ключ=mean всех слоёв,
                  значение=mid, запрос=last (мисматч пространств, «как было»);
* latent_bank   — LatentRagBank: ключ=norm(q_сигнатура+β·тело) на mid,
                  запрос=latent_mean+α·TTL-gist (латентный банк, вариация 3);
* hier_flat     — HierLatentBank mode=flat (вчерашний лидер taskq_large);
* kv_only       — БЕЗ текстовых notes: слот-банк кормится латентными mean
                  тел функций проекта, K/V-инъекция в mid-attention при генерации;
* hier_flat_kv  — hier_flat notes + K/V-инъекция (добавляет ли инжект к поиску).

K/V-банк: один слот на функцию (mid-mean тела) — функции покрывают весь
проект (392 шт ≤ 512 слотов); per-token кормёжка забила бы слоты первыми
файлами. Ингестия тел — смягчённый gate (как в LatentRagBank для кода),
во время генерации corr работает с диалоговым gate (window_crit=10).

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/code_search_retest.py
Выход:  results/chunk_code/search_retest/{summary.json,REPORT.md,<cond>.json}
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
from tmem.chunkmem import ChunkBank, chunk_text, embed_block_key, embed_chunk  # noqa: E402
from tmem.generate import decode_causal  # noqa: E402
from tmem.hiermem import HierLatentBank, build_project_tree  # noqa: E402
from tmem.kvcache import KVConfig  # noqa: E402
from tmem.kvinject import install_kv_memory, uninstall_kv_memory  # noqa: E402
from tmem.latentrag import LatentRagBank, latent_mean  # noqa: E402
from tmem.memory import CorrectionLayer  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.semantic import embed_utterance  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "chunk_code" / "search_retest"
BETA, ALPHA, K_FLAT = 1.0, 0.5, 6
CONDITIONS = ["no_memory", "flat_last", "flat_mid", "flat_legacy",
              "latent_bank", "hier_flat", "kv_only", "hier_flat_kv"]


def request_proxy(node) -> str:
    """Сигнатура + docstring функции — прокси задачи, породившей тело."""
    src = node.text
    try:
        doc = ast.get_docstring(ast.parse(src.lstrip()))
    except SyntaxError:
        doc = None
    first = src.strip().splitlines()[0]
    return first + (" — " + doc if doc else "")


def build_flat_bank(bundle, files: list[str], key_layers: str, val_layer: str) -> tuple[ChunkBank, float]:
    """Плоский банк чанков проекта: ключ=активации блока, значение=span в контексте."""
    bank = ChunkBank(bundle.hidden_size, capacity=8192, centered=True, device=bundle.device)
    t0 = time.perf_counter()
    n_chunks = 0
    for text in files:
        chunks = chunk_text(text, max_tokens=256, tokenizer=bundle.tokenizer)
        for i, ch in enumerate(chunks):
            ctx = "\n".join(chunks[max(0, i - 2):i])
            key, _ = embed_block_key(bundle, ch, layers=key_layers, max_tokens=256)
            val, _, _ = embed_chunk(bundle, ch, context=ctx, mode="span", max_tokens=256,
                                    context_tokens=512, layer=val_layer)
            bank.add(ch, key_emb=key, val_emb=val, turn=n_chunks)
            n_chunks += 1
    return bank, time.perf_counter() - t0


def build_kv_corr(bundle, funcs) -> tuple[CorrectionLayer, float]:
    """Слот-банк для K/V-инъекции: один латентный mean на функцию проекта."""
    corr = CorrectionLayer(bundle.hidden_size, size=512, retrieve_count=256,
                           window_crit=10.0, crit=0.95, max_to_new=0.4, merge_k=0.25)
    t0 = time.perf_counter()
    for n in funcs:
        vec = latent_mean(bundle, n.text, max_tokens=256)
        # код-активации не проходят диалоговый activity-gate — смягчаем,
        # как в LatentRagBank/HierLatentBank при ингестии тел
        corr.memory.update(vec.view(1, 1, -1), crit=0.95, max_to_new=1.0, window_crit=-1e9)
    return corr, time.perf_counter() - t0


def generate(bundle, ctx: str, notes: list[str], task: str) -> str:
    nb = ("Relevant notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n") if notes else ""
    prompt = bundle.to_chat(f"Dialogue.\n{nb}{ctx}\nUser: {task}\nAssistant:")
    ids = bundle.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)["input_ids"]
    ids = ids[:, -bundle.max_prompt_len(reserve=192):].to(bundle.device)
    dec = decode_causal(bundle.model, ids, kv=KVConfig.classic(), cache="use",
                        max_new_tokens=192, do_sample=False, eos_ids=bundle.eos_ids(), seed=0)
    return bundle.tokenizer.decode(dec.token_ids, skip_special_tokens=True).strip()


def score_entry(task: dict, answer: str, notes: list[str]) -> dict:
    try:
        ast.parse(answer if "def " in answer else "pass")
        valid = True
    except SyntaxError:
        valid = False
    return {"task": task["id"], "valid_code": valid,
            "reuse": {n: (n in answer) for n in task["expected"]},
            "retrieval_hit": {n: any((f"def {n}" in note) or (f"{n}(" in note)
                                     for note in notes) for n in task["expected"]},
            "answer": answer, "injected": notes}


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    files = [p.read_text(encoding="utf-8") for p in sorted(PROJECT.rglob("*.py"))]
    nodes = build_project_tree(PROJECT)
    funcs = [n for n in nodes if n.kind == "function"]
    ctx = project_context()
    print(f"проект: {len(files)} файлов, {len(funcs)} функций", flush=True)

    ingest: dict = {}
    # --- плоские банки (три схемы эмбеддингов) ---
    flat_last, ingest["flat_last_s"] = build_flat_bank(bundle, files, "last", "last")
    print(f"flat_last: {len(flat_last.texts)} чанков за {ingest['flat_last_s']:.0f}s", flush=True)
    flat_mid, ingest["flat_mid_s"] = build_flat_bank(bundle, files, "mid", "mid")
    print(f"flat_mid: {len(flat_mid.texts)} чанков за {ingest['flat_mid_s']:.0f}s", flush=True)
    flat_legacy, ingest["flat_legacy_s"] = build_flat_bank(bundle, files, "mean", "mid")
    print(f"flat_legacy: {len(flat_legacy.texts)} чанков за {ingest['flat_legacy_s']:.0f}s", flush=True)

    # --- латентный банк (вариация 3) ---
    latent = LatentRagBank(bundle.hidden_size, beta=BETA, capacity=8192, device=bundle.device)
    t0 = time.perf_counter()
    for n in funcs:
        parent = nodes[n.parent] if n.parent is not None else None
        latent.add(bundle, request_proxy(n), n.text,
                   context=parent.text if parent else "", turn=n.nid)
    ingest["latent_bank_s"] = time.perf_counter() - t0
    print(f"latent_bank: {latent.stats} за {ingest['latent_bank_s']:.0f}s", flush=True)

    # --- дерево + латентные ключи (вчерашний лидер) ---
    hier = HierLatentBank(nodes, beta=BETA, device=bundle.device)
    t0 = time.perf_counter()
    hier.ingest(bundle)
    ingest["hier_s"] = time.perf_counter() - t0
    print(f"hier: {hier.stats} за {ingest['hier_s']:.0f}s", flush=True)

    summary = {"model": "qwen35-4b", "device": device, "project": "taskq_large",
               "n_files": len(files), "n_functions": len(funcs),
               "beta": BETA, "alpha": ALPHA, "k_flat": K_FLAT,
               "ingest_s": {k: round(v, 1) for k, v in ingest.items()},
               "conditions": {}}

    def retrieve_notes(cond: str, task: dict) -> list[str]:
        q = task["user"]
        if cond in ("flat_last", "flat_mid", "flat_legacy"):
            bank = {"flat_last": flat_last, "flat_mid": flat_mid, "flat_legacy": flat_legacy}[cond]
            layer = "mid" if cond == "flat_mid" else "last"
            return [t for t, _, _ in bank.retrieve(embed_utterance(bundle, q, layer=layer),
                                                   k=K_FLAT, score="key")]
        if cond == "latent_bank":
            return [t for t, _, _ in latent.retrieve(bundle, q, k=K_FLAT, alpha=ALPHA)]
        if cond in ("hier_flat", "hier_flat_kv"):
            res = hier.retrieve(bundle, q, k_branches=5, k_children=2, alpha=ALPHA, mode="flat")
            return [n.text for n, _, _ in res]
        return []  # no_memory, kv_only

    for cond in CONDITIONS:
        print(f"\n=== {cond} ===", flush=True)
        t0 = time.perf_counter()
        kv_stats = None
        corr = None
        wrappers = None
        if cond in ("kv_only", "hier_flat_kv"):
            corr, kv_feed_s = build_kv_corr(bundle, funcs)
            ingest[f"{cond}_kv_feed_s"] = round(kv_feed_s, 1)
            wrappers = install_kv_memory(bundle.model, corr, max_slots=32)
            print(f"  kv-банк: {corr.memory.count_element} слотов за {kv_feed_s:.0f}s", flush=True)
        entries = []
        try:
            for task in TASKS:
                notes = retrieve_notes(cond, task)
                answer = generate(bundle, ctx, notes, task["user"])
                entry = score_entry(task, answer, notes)
                entries.append(entry)
                print(f"  {task['id']}: reuse={entry['reuse']} hits={entry['retrieval_hit']}", flush=True)
        finally:
            if wrappers is not None:
                uninstall_kv_memory(bundle.model)
                kv_stats = {"injected_slots": sum(w.stats["injected_slots"] for w in wrappers),
                            "steps": sum(w.stats["steps"] for w in wrappers),
                            "slots_left": corr.memory.count_element}
        summary["conditions"][cond] = {
            "wall_s": round(time.perf_counter() - t0, 1), "kv": kv_stats, "entries": entries}
        (DIR / f"{cond}.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = [
        "# Ретест поисков по коду (taskq_large): латент vs hidden[-1], нужен ли K/V-инжект",
        "",
        f"Проект: {s['n_files']} файлов, {s['n_functions']} функций. 5 задач × 2 хелпера = 10 бинарных исходов.",
        f"Ингестия (с): {s['ingest_s']}.",
        "reuse — имя хелпера в ответе; retrieval hit — def/вызов в notes; valid — синтаксис.",
        "",
        "| Условие | Reuse /10 | Retrieval hits /10 | Valid /5 | Время, с |",
        "|---|---|---|---|---|",
    ]
    for cond, c in s["conditions"].items():
        ents = c["entries"]
        reuse = sum(sum(e["reuse"].values()) for e in ents)
        hits = sum(sum(e["retrieval_hit"].values()) for e in ents)
        valid = sum(e["valid_code"] for e in ents)
        lines.append(f"| {cond} | {reuse} | {hits} | {valid} | {c['wall_s']} |")
    lines += ["", "## По задачам (reuse)", "",
              "| Условие | " + " | ".join(t["id"] for t in TASKS) + " |",
              "|---|" + "---|" * len(TASKS)]
    for cond, c in s["conditions"].items():
        cells = []
        for e in c["entries"]:
            cells.append(", ".join(f"{k}={'да' if v else 'нет'}" for k, v in e["reuse"].items()))
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    kv_conds = {k: v["kv"] for k, v in s["conditions"].items() if v.get("kv")}
    if kv_conds:
        lines += ["", "## K/V-инъекция", ""]
        for cond, kv in kv_conds.items():
            lines.append(f"- {cond}: {kv}")
    lines += ["",
              "flat_last/flat_mid/flat_legacy — плоский ChunkBank (ключ=активации блока, centered);",
              "latent_bank — LatentRagBank (ключ=norm(q_сигнатура+β·тело), mid, TTL-gist запрос);",
              "hier_flat — HierLatentBank mode=flat; kv_only — только K/V-инъекция (слот=mid-mean функции);",
              "hier_flat_kv — hier_flat + K/V-инъекция. Данные: `summary.json`, `<cond>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

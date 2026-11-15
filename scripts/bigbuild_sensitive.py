"""Чувствительная большая сборка + статистика (заказ 15.11, вариант A).

bigbuild показал floor-эффект (модель пишет standalone-код в 90% случаев, reuse
2 vs 6 из 60). Здесь: (1) сильная reuse-инструкция (БЕЗ утечки имён целей) —
поднять baseline reuse выше floor; (2) 5 сидов с sampling для парной статистики.
Гипотеза: living_memctx > static на масштабе, устойчиво по сидам.

36 шагов (core→service→pipeline), window=1, AST reuse/rewrite/miss.
Условия: static (обычные эмбеддинги чанков) vs living_memctx (контекстуализация
K/V-инжектом живой памяти при кодировании + TTL-эволюция + gist).
Парная статистика: mean±std reuse, t-тест living−static, по фазам.

Выход: results/bigbuild_sens/{summary.json,REPORT.md,<cond>_s<seed>.json}
Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/bigbuild_sensitive.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from bigbuild_test import STEPS, make_request, score_reuse  # noqa: E402
from tmem.chunkmem import ChunkBank, chunk_text, embed_block_key, embed_chunk  # noqa: E402
from tmem.generate import decode_causal  # noqa: E402
from tmem.kvcache import KVConfig  # noqa: E402
from tmem.memory import CorrectionLayer  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.semantic import embed_utterance  # noqa: E402
from tmem.semtree import memory_injected  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "bigbuild_sens"
K, WINDOW, ALPHA = 5, 1, 0.5
SEEDS = [0, 1, 2, 3, 4]
TEMP = 0.6
BASE = ("Project: a data-processing platform. CRITICAL RULE: build it by REUSING the "
        "functions you have already written — CALL them directly, do NOT reimplement or "
        "redefine existing functionality. Keep the overall goal of a coherent end-to-end "
        "processing pipeline. Your earlier code is provided; use it.")
CONDITIONS = ["static", "living_memctx"]


def gen(bundle, prompt, seed):
    p = bundle.to_chat(prompt)
    ids = bundle.tokenizer(p, return_tensors="pt", truncation=True, max_length=8192)["input_ids"]
    ids = ids[:, -bundle.max_prompt_len(reserve=256):].to(bundle.device)
    dec = decode_causal(bundle.model, ids, kv=KVConfig.classic(), cache="use",
                        max_new_tokens=256, do_sample=True, temperature=TEMP, top_p=0.9,
                        eos_ids=bundle.eos_ids(), seed=seed)
    return bundle.tokenizer.decode(dec.token_ids, skip_special_tokens=True).strip()


def run_condition(bundle, cond, seed):
    bank = ChunkBank(bundle.hidden_size, capacity=2048, centered=True, device=bundle.device)
    living = cond == "living_memctx"
    corr = CorrectionLayer(bundle.hidden_size, size=512, retrieve_count=256,
                           window_crit=-1e9, crit=0.95, max_to_new=1.0, merge_k=0.25) if living else None
    history, steps_res = [], []
    reuse_n = rewrite_n = miss_n = total = 0
    phase_r = {"core": [0, 0], "service": [0, 0], "pipeline": [0, 0]}
    for i, (name, desc, should) in enumerate(STEPS):
        phase = "core" if i < 12 else "service" if i < 24 else "pipeline"
        req = make_request(name, desc)
        notes = []
        if bank.texts:
            if living:
                with memory_injected(bundle, corr):
                    q = embed_utterance(bundle, req, layer="mean")
                g = corr.memory.ttl_gist()
                if g is not None:
                    q = q + ALPHA * g.to(q.device)
            else:
                q = embed_utterance(bundle, req, layer="mean")
            notes = [t for t, _, _ in bank.retrieve(q, k=K, score="key")]
        nb = ("Your earlier code (CALL these functions, do NOT rewrite them):\n"
              + "\n".join(notes) + "\n") if notes else ""
        recent = "\n".join(history[-WINDOW:]) if history else ""
        prompt = f"Dialogue.\n{BASE}\n{nb}{recent}\nUser: {req}\nAssistant:"
        answer = gen(bundle, prompt, seed + i)
        sc = score_reuse(answer, should)
        for v in sc.values():
            total += 1
            reuse_n += v == "reuse"; rewrite_n += v == "rewrite"; miss_n += v == "miss"
        if should:
            phase_r[phase][0] += sum(v == "reuse" for v in sc.values())
            phase_r[phase][1] += len(should)
        steps_res.append({"i": i, "phase": phase, "step": name, "score": sc})
        for ch in chunk_text(answer, max_tokens=256, tokenizer=bundle.tokenizer):
            if living:
                with memory_injected(bundle, corr):
                    val, toks, _ = embed_chunk(bundle, ch, mode="late", layer="mid", max_tokens=256)
                    key, _ = embed_block_key(bundle, ch, layers="mean", max_tokens=256)
                if toks is not None:
                    corr.memory.update(toks.to(bundle.device), crit=0.95, max_to_new=1.0, window_crit=-1e9)
            else:
                val, toks, _ = embed_chunk(bundle, ch, mode="late", layer="mid", max_tokens=256)
                key, _ = embed_block_key(bundle, ch, layers="mean", max_tokens=256)
            bank.add(ch, key_emb=key, val_emb=val)
        history.append(f"User: {req}\nAssistant: {answer}")
    return {"seed": seed, "reuse": reuse_n, "rewrite": rewrite_n, "miss": miss_n, "total": total,
            "phase": {p: f"{r}/{t}" for p, (r, t) in phase_r.items()}, "steps": steps_res}


def stats(deltas):
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    t = mean / (std / math.sqrt(n)) if std > 0 else (float("inf") if mean else 0.0)
    return {"mean": round(mean, 2), "std": round(std, 2), "t": round(t, 3), "df": n - 1,
            "pos": sum(d > 0 for d in deltas), "neg": sum(d < 0 for d in deltas),
            "zero": sum(d == 0 for d in deltas)}


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    summary = {"model": "qwen35-4b", "device": device, "n_steps": len(STEPS),
               "seeds": SEEDS, "temp": TEMP, "conditions": {}}
    runs = {c: [] for c in CONDITIONS}
    for cond in CONDITIONS:
        for seed in SEEDS:
            t0 = time.perf_counter()
            res = run_condition(bundle, cond, seed)
            res["wall_s"] = round(time.perf_counter() - t0, 1)
            runs[cond].append(res)
            (DIR / f"{cond}_s{seed}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  {cond} seed{seed}: reuse {res['reuse']}/{res['total']} "
                  f"phase {res['phase']} ({res['wall_s']}s)", flush=True)
    for cond in CONDITIONS:
        rs = runs[cond]
        summary["conditions"][cond] = {
            "reuse_mean": round(sum(r["reuse"] for r in rs) / len(rs), 2),
            "reuse_std": round(math.sqrt(sum((r["reuse"] - sum(x["reuse"] for x in rs)/len(rs))**2 for r in rs)/(len(rs)-1)), 2),
            "total": rs[0]["total"], "per_seed": [r["reuse"] for r in rs],
            "phase_mean": {p: round(sum(int(r["phase"][p].split("/")[0]) for r in rs)/len(rs), 2)
                           for p in ("core", "service", "pipeline")}}
    summary["paired_living_minus_static"] = stats([l["reuse"] - s["reuse"] for l, s in zip(runs["living_memctx"], runs["static"])])
    (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = ["# Чувствительная большая сборка + статистика (5 сидов)", "",
             f"{s['n_steps']} шагов, temp={s['temp']}, сиды {s['seeds']}. Сильная reuse-инструкция.", "",
             "| Условие | reuse mean±std | core | service | pipeline | по сидам |",
             "|---|---|---|---|---|---|"]
    for cond, c in s["conditions"].items():
        pm = c["phase_mean"]
        lines.append(f"| {cond} | {c['reuse_mean']}±{c['reuse_std']} /{c['total']} "
                     f"| {pm['core']} | {pm['service']} | {pm['pipeline']} | {c['per_seed']} |")
    p = s["paired_living_minus_static"]
    lines += ["", f"## Парный эффект living−static: mean {p['mean']} ± {p['std']}, "
              f"t={p['t']} (df={p['df']}), +/{p['pos']} −/{p['neg']} =/{p['zero']}",
              "t-крит df=4 → 2.776 (α=0.05 двусторонний). |t|>крит → значимо.", "",
              "Гипотеза: living_memctx > static устойчиво по сидам. Данные: `<cond>_s<seed>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

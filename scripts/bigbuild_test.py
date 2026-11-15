"""Большая сборка проекта: живая память удерживает направление на масштабе.

Гипотеза автора: на ОГРОМном проекте статический поиск захлёбывается, а
контекстуализация чанков через активировавшиеся слои латентной памяти + эволюция
памяти (TTL/merge) помогают следовать общей цели и поддерживать направление
написания кода. Смотрим кривую reuse по мере роста кодовой базы.

36 шагов: 12 core-утилит → 12 сервисов (reuse 2 core) → 12 пайплайнов (reuse
2 service + 1 core). Запросы НЕ называют reuse-цели — агент сам вспоминает свой
код. window=1. AST-скоринг reuse/rewrite/miss._REUSE-цели поздних шагов погребены
под десятками prior-функций → статический поиск должен шуметь.

Условия:
* static        — ChunkBank, обычные эмбеддинги чанков, косинус-поиск;
* living_memctx — чанки контекстуализируются K/V-инжектом живой памяти при
                  кодировании (memory_injected) + TTL-банк + gist-запрос.

Выход: results/bigbuild/{summary.json,REPORT.md,<cond>.json}
Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/bigbuild_test.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from tmem.chunkmem import ChunkBank, chunk_text, embed_block_key, embed_chunk  # noqa: E402
from tmem.generate import decode_causal  # noqa: E402
from tmem.kvcache import KVConfig  # noqa: E402
from tmem.memory import CorrectionLayer  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.semantic import embed_utterance  # noqa: E402
from tmem.semtree import memory_injected  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "bigbuild"
K, WINDOW, ALPHA = 5, 1, 0.5

# (name, description, reuse_targets) — запросы без имён reuse-целей
CORE = [
    ("validate_record", "checks a record has required fields id and amount", []),
    ("normalize_amount", "normalizes the amount field to a safe numeric range", []),
    ("dedupe_by_id", "removes duplicate records by id keeping the first", []),
    ("parse_timestamp", "parses a timestamp string into epoch milliseconds", []),
    ("format_currency", "formats a numeric amount as a currency string", []),
    ("clamp_value", "clamps a value to a safe min/max range", []),
    ("retry_operation", "retries a failing operation with exponential backoff", []),
    ("cache_lookup", "caches a computed result with a time-to-live", []),
    ("log_event", "writes a structured log event with a level and message", []),
    ("compute_metric", "aggregates a numeric metric over records", []),
    ("slugify_name", "converts a name into a URL-safe slug", []),
    ("chunk_list", "splits a list into fixed-size batches", []),
]
SERVICES = [
    ("ingest_records", "validates and normalizes incoming records", ["validate_record", "normalize_amount"]),
    ("clean_records", "removes duplicates and clamps values of records", ["dedupe_by_id", "clamp_value"]),
    ("enrich_records", "parses timestamps and formats currency for records", ["parse_timestamp", "format_currency"]),
    ("resilient_call", "calls an operation with retry and caching", ["retry_operation", "cache_lookup"]),
    ("audit_trail", "logs events and computes an audit metric", ["log_event", "compute_metric"]),
    ("name_records", "slugifies record names after validating them", ["slugify_name", "validate_record"]),
    ("batch_records", "splits records into batches and dedupes each", ["chunk_list", "dedupe_by_id"]),
    ("time_series", "parses timestamps and computes a time metric", ["parse_timestamp", "compute_metric"]),
    ("money_report", "formats currency and computes a money total", ["format_currency", "compute_metric"]),
    ("safe_transform", "clamps and normalizes record values", ["clamp_value", "normalize_amount"]),
    ("cached_ingest", "caches and validates incoming records", ["cache_lookup", "validate_record"]),
    ("logged_batch", "logs and batches records", ["log_event", "chunk_list"]),
]
PIPELINES = [
    ("etl_pipeline", "ingests then cleans records and computes a metric", ["ingest_records", "clean_records", "compute_metric"]),
    ("report_pipeline", "enriches records and produces a money report", ["enrich_records", "money_report", "format_currency"]),
    ("resilient_etl", "resiliently ingests records with retry", ["resilient_call", "ingest_records", "retry_operation"]),
    ("audit_pipeline", "cleans records and writes an audit trail", ["audit_trail", "clean_records", "log_event"]),
    ("naming_pipeline", "names records then batches them", ["name_records", "batch_records", "slugify_name"]),
    ("timeseries_pipeline", "builds a time series from enriched records", ["time_series", "enrich_records", "parse_timestamp"]),
    ("safe_etl", "safely transforms then ingests records", ["safe_transform", "ingest_records", "clamp_value"]),
    ("cached_pipeline", "cached ingestion followed by cleaning", ["cached_ingest", "clean_records", "cache_lookup"]),
    ("logged_pipeline", "logged batching followed by an audit trail", ["logged_batch", "audit_trail", "log_event"]),
    ("money_pipeline", "produces a money report from cleaned records", ["money_report", "clean_records", "compute_metric"]),
    ("full_pipeline", "ingests, cleans, and reports records end to end", ["etl_pipeline", "report_pipeline", "compute_metric"]),
    ("resilient_full", "resilient end-to-end processing with retry and audit", ["resilient_etl", "audit_pipeline", "retry_operation"]),
]
STEPS = CORE + SERVICES + PIPELINES
CONDITIONS = ["static", "living_memctx"]


def make_request(name, desc):
    return f"Write a function {name}(records) that {desc}. Reply with code only."


def score_reuse(answer: str, should: list[str]) -> dict:
    code = answer
    m = re.search(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
    if m:
        code = m.group(1)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {n: "miss" for n in should}
    called, defined = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
    out = {}
    for n in should:
        if n in defined:
            out[n] = "rewrite"
        elif n in called:
            out[n] = "reuse"
        else:
            out[n] = "miss"
    return out


def gen(bundle, prompt):
    p = bundle.to_chat(prompt)
    ids = bundle.tokenizer(p, return_tensors="pt", truncation=True, max_length=8192)["input_ids"]
    ids = ids[:, -bundle.max_prompt_len(reserve=256):].to(bundle.device)
    dec = decode_causal(bundle.model, ids, kv=KVConfig.classic(), cache="use",
                        max_new_tokens=256, do_sample=False, eos_ids=bundle.eos_ids(), seed=0)
    return bundle.tokenizer.decode(dec.token_ids, skip_special_tokens=True).strip()


def run_condition(bundle, cond, base):
    bank = ChunkBank(bundle.hidden_size, capacity=2048, centered=True, device=bundle.device)
    corr = CorrectionLayer(bundle.hidden_size, size=512, retrieve_count=256,
                           window_crit=-1e9, crit=0.95, max_to_new=1.0, merge_k=0.25) if cond == "living_memctx" else None
    living = cond == "living_memctx"
    history, steps_res = [], []
    reuse_n = rewrite_n = miss_n = total = 0
    for i, (name, desc, should) in enumerate(STEPS):
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
        nb = ("Your earlier code:\n" + "\n".join(notes) + "\n") if notes else ""
        recent = "\n".join(history[-WINDOW:]) if history else ""
        prompt = f"Dialogue.\n{base}\n{nb}{recent}\nUser: {req}\nAssistant:"
        answer = gen(bundle, prompt)
        sc = score_reuse(answer, should)
        for v in sc.values():
            total += 1
            reuse_n += v == "reuse"; rewrite_n += v == "rewrite"; miss_n += v == "miss"
        steps_res.append({"i": i, "phase": "core" if i < 12 else "service" if i < 24 else "pipeline",
                          "step": name, "should": should, "score": sc,
                          "cum_reuse": reuse_n, "cum_total": total})
        # ингестия своего кода
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
        if should:
            print(f"  [{cond}] {i:2d} {name}: {sc} (cum reuse {reuse_n}/{total})", flush=True)
    return {"reuse": reuse_n, "rewrite": rewrite_n, "miss": miss_n, "total": total,
            "bank_size": len(bank.texts), "steps": steps_res}


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    base = ("Project: a data-processing platform. Build it incrementally, reusing the "
            "functions you have already written wherever they fit. Keep the overall goal "
            "of a coherent end-to-end processing pipeline.")
    summary = {"model": "qwen35-4b", "device": device, "n_steps": len(STEPS),
               "window": WINDOW, "k": K, "conditions": {}}
    for cond in CONDITIONS:
        t0 = time.perf_counter()
        print(f"\n=== {cond} ===", flush=True)
        res = run_condition(bundle, cond, base)
        res["wall_s"] = round(time.perf_counter() - t0, 1)
        # reuse по фазам
        for phase in ("core", "service", "pipeline"):
            ps = [s for s in res["steps"] if s["phase"] == phase and s["should"]]
            r = sum(sum(v == "reuse" for v in s["score"].values()) for s in ps)
            t = sum(len(s["should"]) for s in ps)
            res[f"{phase}_reuse"] = f"{r}/{t}"
        summary["conditions"][cond] = res
        (DIR / f"{cond}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {cond} ИТОГ: reuse {res['reuse']}/{res['total']} rewrite {res['rewrite']} "
              f"miss {res['miss']} | core {res['core_reuse']} service {res['service_reuse']} "
              f"pipeline {res['pipeline_reuse']} ({res['wall_s']}s)", flush=True)
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = ["# Большая сборка: живая память на масштабе (36 шагов)", "",
             f"window={s['window']}, k={s['k']}. reuse=вызов prior-функции; rewrite=переопределение; miss=забыл.",
             "Гипотеза: static деградирует с ростом базы, living_memctx удерживает направление.", "",
             "| Условие | reuse | rewrite | miss | core | service | pipeline | банк |",
             "|---|---|---|---|---|---|---|---|"]
    for cond, c in s["conditions"].items():
        lines.append(f"| {cond} | {c['reuse']}/{c['total']} | {c['rewrite']} | {c['miss']} "
                     f"| {c['core_reuse']} | {c['service_reuse']} | {c['pipeline_reuse']} | {c['bank_size']} |")
    lines += ["", "## Кривая cumulative reuse (по шагам с reuse-целями)", "",
              "| шаг | фаза | static cum | living cum |", "|---|---|---|---|"]
    sc = {s["i"]: s for s in s["conditions"]["static"]["steps"]}
    lc = {s["i"]: s for s in s["conditions"]["living_memctx"]["steps"]}
    for i in sorted(sc):
        if sc[i]["should"]:
            lines.append(f"| {i} {sc[i]['step']} | {sc[i]['phase']} "
                         f"| {sc[i]['cum_reuse']}/{sc[i]['cum_total']} | {lc[i]['cum_reuse']}/{lc[i]['cum_total']} |")
    lines += ["", "static — обычные эмбеддинги чанков; living_memctx — контекстуализация K/V-инжектом",
              "живой памяти при кодировании + TTL-эволюция + gist. Данные: `<cond>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

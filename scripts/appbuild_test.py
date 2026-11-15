"""App-building reuse-vs-write тест (заказ 15.11): живая память помнит СВОЙ код.

Гипотеза автора: живая память удерживает целостную модель приложения и помогает
понять, где переиспользовать существующее (в т.ч. СВОЙ ранний код), а где писать
новое. Последовательная сборка: агент пишет функции на ранних шагах и должен
переиспользовать их на поздних. Короткое окно (window=1) форсирует опору на
память (поздний шаг не видит ранний код напрямую).

Шаги (must_reuse = что обязательно переиспользовать; own = свой ранний код):
1 rate_limiter (new)        reuse: clamp [base]
2 cache_result (new)        reuse: now_ms [base]
3 guarded_call              reuse: rate_limiter, cache_result [OWN]
4 batch_guarded             reuse: guarded_call [OWN], chunked [base]
5 invoice_guarded           reuse: guarded_call [OWN], summarize [base]

Условия:
* no_memory — только окно (без поиска раннего кода);
* static    — ChunkBank: поиск своего кода по косинусу;
* living    — ChunkBank + TTL-банк: запрос кондиционируется gist живой памяти.

Метрика: reuse correctness (имена must_reuse в коде), valid syntax.
Выход: results/appbuild/{summary.json,REPORT.md,<cond>.json}
Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/appbuild_test.py
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

from taskq_large_eval import project_context  # noqa: E402
from tmem.chunkmem import ChunkBank, chunk_text, embed_block_key, embed_chunk  # noqa: E402
from tmem.generate import decode_causal  # noqa: E402
from tmem.kvcache import KVConfig  # noqa: E402
from tmem.memory import Memory  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.semantic import embed_utterance  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "appbuild"
WINDOW, K, ALPHA = 1, 4, 0.5
STEPS = [
    {"id": "rate_limiter", "must_reuse": ["clamp"], "own": [],
     "req": "Write a function rate_limiter(max_calls, period) that limits the call rate, "
            "clamping values to a safe range with the project's clamp helper. Reply with code only."},
    {"id": "cache_result", "must_reuse": ["now_ms"], "own": [],
     "req": "Write a function cache_result(key, ttl) that caches a result with a TTL using the "
            "project's now_ms helper. Reply with code only."},
    {"id": "guarded_call", "must_reuse": ["rate_limiter", "cache_result"], "own": ["rate_limiter", "cache_result"],
     "req": "Write a function guarded_call(fn) that rate-limits and caches calls to fn. "
            "REUSE the rate_limiter and cache_result functions you already wrote. Reply with code only."},
    {"id": "batch_guarded", "must_reuse": ["guarded_call", "chunked"], "own": ["guarded_call"],
     "req": "Write a function batch_guarded(items) that applies guarded_call to batches of items, "
            "REUSING guarded_call you wrote and the project's chunked helper. Reply with code only."},
    {"id": "invoice_guarded", "must_reuse": ["guarded_call", "summarize"], "own": ["guarded_call"],
     "req": "Write a service method process_invoices(records) that processes invoices with rate "
            "limiting and caching, REUSING guarded_call you wrote and the project's summarize. Reply with code only."},
]
CONDITIONS = ["no_memory", "static", "living"]


def gen(bundle, prompt: str) -> str:
    p = bundle.to_chat(prompt)
    ids = bundle.tokenizer(p, return_tensors="pt", truncation=True, max_length=8192)["input_ids"]
    ids = ids[:, -bundle.max_prompt_len(reserve=256):].to(bundle.device)
    dec = decode_causal(bundle.model, ids, kv=KVConfig.classic(), cache="use",
                        max_new_tokens=256, do_sample=False, eos_ids=bundle.eos_ids(), seed=0)
    return bundle.tokenizer.decode(dec.token_ids, skip_special_tokens=True).strip()


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    base = project_context()
    summary = {"model": "qwen35-4b", "device": device, "window": WINDOW, "conditions": {}}

    for cond in CONDITIONS:
        print(f"\n=== {cond} ===", flush=True)
        bank = ChunkBank(bundle.hidden_size, capacity=256, centered=True, device=bundle.device)
        slots = Memory(bundle.hidden_size, size=256, merge_k=0.25, device=bundle.device)
        history: list[str] = []
        entries = []
        for step in STEPS:
            notes = []
            if cond != "no_memory" and bank.texts:
                q = embed_utterance(bundle, step["req"], layer="last")
                if cond == "living":
                    slots.get_relevant(q.view(1, 1, -1))
                    g = slots.ttl_gist()
                    if g is not None:
                        q = q + ALPHA * g.to(q.device)
                notes = [t for t, _, _ in bank.retrieve(q, k=K, score="key")]
            nb = ("Your earlier code:\n" + "\n".join(notes) + "\n") if notes else ""
            recent = "\n".join(history[-WINDOW:]) if history else ""
            prompt = f"Dialogue.\n{base}\n{nb}{recent}\nUser: {step['req']}\nAssistant:"
            answer = gen(bundle, prompt)
            try:
                ast.parse(answer if "def " in answer else "pass")
                valid = True
            except SyntaxError:
                valid = False
            reuse = {n: (n in answer) for n in step["must_reuse"]}
            own_reuse = {n: (n in answer) for n in step["own"]}
            entry = {"step": step["id"], "valid": valid, "reuse": reuse, "own_reuse": own_reuse,
                     "retrieved": [n.splitlines()[0][:50] if n else "" for n in notes],
                     "answer": answer}
            entries.append(entry)
            print(f"  {step['id']}: reuse={reuse} own={own_reuse} valid={valid}", flush=True)
            # ингестия своего кода в банк(и)
            for ch in chunk_text(answer, max_tokens=256, tokenizer=bundle.tokenizer):
                key, _ = embed_block_key(bundle, ch, layers="mean", max_tokens=256)
                val, toks, _ = embed_chunk(bundle, ch, mode="late", layer="mid", max_tokens=256)
                bank.add(ch, key_emb=key, val_emb=val)
                if cond == "living" and toks is not None:
                    slots.update(toks.to(bundle.device), crit=0.95, max_to_new=1.0, window_crit=-1e9)
            history.append(f"User: {step['req']}\nAssistant: {answer}")
        reuse_total = sum(sum(e["reuse"].values()) for e in entries)
        reuse_max = sum(len(e["reuse"]) for e in entries)
        own_total = sum(sum(e["own_reuse"].values()) for e in entries)
        own_max = sum(len(e["own_reuse"]) for e in entries)
        valid_n = sum(e["valid"] for e in entries)
        summary["conditions"][cond] = {"reuse": f"{reuse_total}/{reuse_max}",
                                       "own_reuse": f"{own_total}/{own_max}",
                                       "valid": f"{valid_n}/{len(entries)}", "entries": entries}
        (DIR / f"{cond}.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        (DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary)
    print("\nготово:", DIR, flush=True)


def write_report(s: dict) -> None:
    lines = ["# App-building: живая память помнит СВОЙ код (reuse-vs-write)", "",
             f"Окно={s['window']}, 5 шагов. own_reuse — переиспользование СВОЕГО раннего кода.", "",
             "| Условие | reuse (всего) | own_reuse (свой код) | valid |", "|---|---|---|---|"]
    for cond, c in s["conditions"].items():
        lines.append(f"| {cond} | {c['reuse']} | {c['own_reuse']} | {c['valid']} |")
    lines += ["", "## По шагам (own_reuse)", "",
              "| Условие | " + " | ".join(st["id"] for st in STEPS) + " |",
              "|---|" + "---|" * len(STEPS)]
    for cond, c in s["conditions"].items():
        cells = [", ".join(f"{k}={'да' if v else 'нет'}" for k, v in e["own_reuse"].items()) or "—"
                 for e in c["entries"]]
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["", "no_memory — только окно; static — ChunkBank-поиск; living — ChunkBank+TTL-gist.",
              "Гипотеза: living лучше помнит свой ранний код (own_reuse) при коротком окне. Данные: `<cond>.json`.", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

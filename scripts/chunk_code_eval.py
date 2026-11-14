"""CPU-оценка чанковой памяти на задаче написания кода (заказ 14.11.2026).

Сценарий: модель пишет функции (turns 0-2), затем идёт filler-болтовня
(turns 3-7), затем probes (turns 8-10) просят написать новый код с
переиспользованием ранее написанных функций. Окно контекста = 2 пары,
поэтому ранний код из контекста выпадает — вернуть его может только память.

Сравниваются:
* window_only — без памяти (базовый);
* window_rag_userfacts — текущая система: банк хранит запросы пользователя;
* chunks_{mean,span,attn} — банк хранит чанки генерации (код), ключ = запрос;
* chunks_span_value — скоринг по содержимому чанка, а не по ключу;
* chunks_late_maxsim — per-token MaxSim.

Метрики: (1) reuse — имя нужной функции в ответе (keyword probe);
(2) retrieval hit — положил ли банк правильный код в промпт (по notes);
(3) латентность/токены. Модель: qwen2-1.5b (base), CPU, float32, greedy.

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/chunk_code_eval.py
Результаты: results/chunk_code/code-v1/ (JSON по стратегиям + summary.json).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

torch.set_num_threads(8)  # ночной GPU-прогон тоже ест CPU — оставляем ему ядра

from tmem.dialogue import Strategy, run_session  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.qa import DialogSession, Probe  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "chunk_code" / "code-v1"

SESSION = DialogSession(
    name="code-reuse",
    opening=[
        "Write a Python function factorial(n) that returns the factorial of n. Reply with code only.",
        "Write a Python function reverse_string(s) that returns s reversed. Reply with code only.",
        "Write a Python function count_vowels(s) that counts vowels in s. Reply with code only.",
    ],
    fillers=[
        "Thanks. Briefly, what is a pure function?",
        "Name one style rule for clean Python code.",
        "What does PEP 8 say about indentation?",
        "Suggest a good name for a variable holding a list of users.",
        "How do you usually debug a wrong result?",
    ],
    probes=[
        Probe(after_turn=8,
              question="Write is_palindrome(s) that checks whether s reads the same backwards. "
                       "Reuse the function you wrote earlier for reversing. Reply with code only.",
              expected=["reverse_string"]),
        Probe(after_turn=9,
              question="Write double_factorial(n) that returns factorial(n) * factorial(n-1). "
                       "Reuse your earlier factorial function. Reply with code only.",
              expected=["factorial"]),
        Probe(after_turn=10,
              question="Write vowel_share(s) returning the fraction of vowels in s. "
                       "Reuse your earlier vowel-counting function. Reply with code only.",
              expected=["count_vowels"]),
    ],
)

# какой def должен вернуться из памяти для каждой пробы
PROBE_TARGET_DEF = {8: "def reverse_string", 9: "def factorial", 10: "def count_vowels"}

STRATEGIES = [
    Strategy("window_only", context="window", window=2),
    Strategy("window_rag_userfacts", context="window", window=2,
             rag=True, rag_centered=True, rag_crit=0.95, rag_lambda=0.0),
    Strategy("chunks_mean", context="window", window=2,
             gen_chunks=True, chunk_key="request", chunk_embed="mean", chunk_score="key"),
    Strategy("chunks_span", context="window", window=2,
             gen_chunks=True, chunk_key="request", chunk_embed="span", chunk_score="key"),
    Strategy("chunks_attn", context="window", window=2,
             gen_chunks=True, chunk_key="request", chunk_embed="attn", chunk_score="key"),
    Strategy("chunks_span_value", context="window", window=2,
             gen_chunks=True, chunk_key="request", chunk_embed="span", chunk_score="value"),
    Strategy("chunks_late_maxsim", context="window", window=2,
             gen_chunks=True, chunk_key="request", chunk_embed="late", chunk_score="maxsim"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("загрузка qwen2-1.5b на CPU (float32, eager attention)...", flush=True)
    bundle = load_model("qwen2-1.5b", device="cpu", dtype=torch.float32, eager=True)
    summary = {"model": "qwen2-1.5b", "device": "cpu", "dtype": "float32",
               "session": SESSION.name, "window": 2, "max_new_tokens": 64,
               "decoding": "greedy", "strategies": {},
               "caveats": ["turn9: имя factorial есть в самом вопросе — возможен echo-ложняк; "
                           "чистые пробы: turn8, turn10",
                           "attn-режиму нужен eager attention (SDPA не отдаёт веса)"]}
    for strategy in STRATEGIES:
        done_path = OUT / f"{strategy.name}.json"
        if done_path.exists():  # resume: готовые стратегии не пересчитываем
            prev = json.loads(done_path.read_text(encoding="utf-8"))
            summary["strategies"][strategy.name] = {
                k: v for k, v in prev.items() if k not in ("turns", "strategy")}
            print(f"\n=== {strategy.name} === уже готов, пропуск", flush=True)
            continue
        t0 = time.perf_counter()
        print(f"\n=== {strategy.name} ===", flush=True)
        run = run_session(bundle, SESSION, strategy, max_new_tokens=64, seed=0)
        wall = time.perf_counter() - t0
        s = run.summary()

        # retrieval hit: банк положил правильный код в промпт пробы?
        hits = {}
        for i, t in enumerate(run.turns):
            if i in PROBE_TARGET_DEF:
                notes = t.injected or []
                hits[str(i)] = any(PROBE_TARGET_DEF[i] in n for n in notes)
        entry = {
            "wall_s": round(wall, 1),
            "probe_mean": s["probe_mean"],
            "probe_scores": s["probe_scores"],
            "retrieval_hits": hits,
            "mean_turn_latency_s": s["mean_turn_latency_s"],
            "embed_tokens": s["embed_tokens"],
            "prompt_tokens": s["prompt_tokens"],
            "chunk_stats": s.get("chunk_stats"),
            "rag_stats": s.get("rag_stats"),
            "turns": [{"turn": i, "user": t.user_text, "answer": t.answer_text,
                       "injected": t.injected, "probe_score": t.probe_score}
                      for i, t in enumerate(run.turns)],
        }
        summary["strategies"][strategy.name] = {k: v for k, v in entry.items() if k != "turns"}
        (OUT / f"{strategy.name}.json").write_text(
            json.dumps({"strategy": strategy.describe(), **entry}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  probe_mean={s['probe_mean']} retrieval_hits={hits} wall={wall:.0f}s", flush=True)
    print("\nготово:", OUT, flush=True)


if __name__ == "__main__":
    main()

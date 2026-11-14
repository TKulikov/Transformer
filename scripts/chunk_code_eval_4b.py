"""Проверка чанковой памяти с КОНТЕНТНЫМ ключом на qwen35-4b (GPU, bf16).

Заказ 14.11: ключ = сам блок + активации слоёв (как hidden states в
чат-версии), значение = контекстуализированный чанк, центрирование бегущим
средним с обеих сторон. Один запрос может породить тысячи строк — контентный
ключ должен позволять вспоминать нужный кусок, а не всю генерацию.

Сетка (та же сессия code-reuse, что в CPU-оценке, для сопоставимости):
* window_only / window_rag_userfacts — контроли;
* chunks_reqkey_span — прежний дизайн (ключ = запрос);
* chunks_contentkey_{span,last,mid} — новый ключ (mean/last/mid слои);
* chunks_contentkey_value — скоринг по значению вместо ключа;
* chunks_late_maxsim — per-token MaxSim.

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/chunk_code_eval_4b.py
Результаты: results/chunk_code/code-4b-v1/ (JSON по стратегиям + summary.json).
Очередь: scripts/queue_chunk_4b.sh ждёт окончания текущего corpus_run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from chunk_code_eval import PROBE_TARGET_DEF, SESSION  # noqa: E402
from tmem.dialogue import Strategy, run_session  # noqa: E402
from tmem.models import load_model  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "chunk_code" / "code-4b-v1"

W = dict(context="window", window=2)
STRATEGIES = [
    Strategy("window_only", **W),
    Strategy("window_rag_userfacts", **W, rag=True, rag_centered=True, rag_crit=0.95, rag_lambda=0.0),
    Strategy("chunks_reqkey_span", **W, gen_chunks=True, chunk_key="request",
             chunk_embed="span", chunk_score="key"),
    Strategy("chunks_contentkey_span", **W, gen_chunks=True, chunk_key="content",
             chunk_key_layers="mean", chunk_embed="span", chunk_score="key"),
    Strategy("chunks_contentkey_last", **W, gen_chunks=True, chunk_key="content",
             chunk_key_layers="last", chunk_embed="span", chunk_score="key"),
    Strategy("chunks_contentkey_mid", **W, gen_chunks=True, chunk_key="content",
             chunk_key_layers="mid", chunk_embed="span", chunk_score="key"),
    Strategy("chunks_contentkey_value", **W, gen_chunks=True, chunk_key="content",
             chunk_key_layers="mean", chunk_embed="span", chunk_score="value"),
    Strategy("chunks_late_maxsim", **W, gen_chunks=True, chunk_key="content",
             chunk_key_layers="mean", chunk_embed="late", chunk_score="maxsim"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device} (bfloat16)...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    summary = {"model": "qwen35-4b", "device": device, "dtype": "bfloat16",
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "torch": torch.__version__,
               "session": SESSION.name, "window": 2, "max_new_tokens": 64,
               "decoding": "greedy", "strategies": {},
               "caveats": ["turn9: имя factorial в вопросе — echo-ложняк; чистые пробы: turn8, turn10",
                           "ключ=активации блока (mean/last/mid), значение=span, центрирование обоих бегущим средним"]}
    for strategy in STRATEGIES:
        done_path = OUT / f"{strategy.name}.json"
        if done_path.exists():
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

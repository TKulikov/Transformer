"""Демо чанковой памяти: агент «вспоминает» свои функции по сходству задач.

Механика (не качество — tiny-модель случайная, качество проверяем позже на
qwen35-4b): поток генерации режется на чанки, каждый эмбеддится в контексте
породившего его запроса и кладётся в ChunkBank с ключом-запросом. Новый
запрос вытаскивает релевантные чанки тремя способами скоринга.

Запуск (CPU, офлайн): .venv/bin/python scripts/demo_chunkmem.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tmem.chunkmem import ChunkBank, chunk_text, embed_chunk  # noqa: E402
from tmem.dialogue import Strategy, run_session  # noqa: E402
from tmem.models import tiny_causal  # noqa: E402
from tmem.qa import DialogSession, Probe  # noqa: E402
from tmem.semantic import embed_utterance  # noqa: E402

# «Поток генерации» агента: задачи и функции, которые он написал.
EPISODES = [
    ("посчитать факториал числа",
     "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"),
    ("развернуть строку задом наперёд",
     "def reverse_string(s):\n    return s[::-1]"),
    ("проверить, является ли слово палиндромом",
     "def is_palindrome(s):\n    return s == s[::-1]"),
    ("найти максимум в списке",
     "def max_item(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best"),
]

NEW_TASK = "нужно проверить, читается ли слово одинаково в обе стороны"


def main() -> None:
    bundle = tiny_causal(d_model=64, layers=2, seed=7)
    bank = ChunkBank(bundle.hidden_size, device=bundle.device)

    print("=== наполнение банка из потока генерации (режим span) ===")
    for task, code in EPISODES:
        context = f"User: {task}\nAssistant:"
        key = embed_utterance(bundle, task)
        for ch in chunk_text(code, max_tokens=128, tokenizer=bundle.tokenizer):
            val, _, n_tok = embed_chunk(bundle, ch, context=context, mode="span")
            bank.add(ch, key_emb=key, val_emb=val, turn=len(bank.texts))
        print(f"  + {task!r} -> чанков: {len(chunk_text(code))}, токенов на эмбеддинг: {n_tok}")
    print(f"банк: {bank.stats}, размер: {len(bank.texts)}")

    print(f"\n=== новый запрос: {NEW_TASK!r} ===")
    q = embed_utterance(bundle, NEW_TASK)
    _, q_tokens, _ = embed_chunk(bundle, NEW_TASK, mode="late")
    for score in ("key", "value", "maxsim"):
        print(f"\nскоринг {score}:")
        for text, turn, s in bank.retrieve(q, k=2, score=score, query_tokens=q_tokens):
            print(f"  [{s:.3f}] (эпизод {turn}) {text.splitlines()[0]}")

    print("\n=== интеграция: run_session c gen_chunks=True (все режимы эмбеддинга) ===")
    session = DialogSession(
        name="chunk-demo",
        opening=["Remember: my helper tool is called Oxidum.", "Tell me about tools."],
        fillers=["Continue."],
        probes=[Probe(after_turn=2, question="What is my helper tool called?", expected=["oxidum"])],
    )
    for mode in ("mean", "span", "attn", "late"):
        strategy = Strategy(f"chunks_{mode}", context="window", window=2,
                            gen_chunks=True, chunk_embed=mode,
                            chunk_score="maxsim" if mode == "late" else "key")
        run = run_session(bundle, session, strategy, max_new_tokens=8, seed=0)
        s = run.summary()
        print(f"  {mode:5s}: turns={s['turns']} chunk_stats={s.get('chunk_stats')} "
              f"embed_tokens={s['embed_tokens']}")

    print("\nготово: механика работает end-to-end; качество — на большой модели позже.")


if __name__ == "__main__":
    main()

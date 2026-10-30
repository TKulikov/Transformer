"""CLI/answer-questions plumbing on tiny offline models (no downloads)."""

from __future__ import annotations

from tmem.cli import answer_questions
from tmem.kvcache import KVConfig
from tmem.qa import GENERAL_QA


def test_answer_questions_causal(causal_bundle):
    out = answer_questions(causal_bundle, kv=KVConfig.classic(), max_new_tokens=6, seed=0)
    assert set(out) == {item.question for item in GENERAL_QA}
    for record in out.values():
        assert 0.0 <= record["score"] <= 1.0
        assert isinstance(record["answer"], str)


def test_answer_questions_sparse_bounds(causal_bundle):
    out = answer_questions(
        causal_bundle,
        kv=KVConfig.sparse(budget=10, strategy="sliding", sink=2, recent=4),
        max_new_tokens=8,
        seed=0,
    )
    assert len(out) == len(GENERAL_QA)

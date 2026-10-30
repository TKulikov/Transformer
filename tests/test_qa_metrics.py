"""Tests for QA scripts, scoring and metric functions."""

from __future__ import annotations

import pytest

from tmem.metrics import keyword_score, novelty_quality, strict_quality, summarize
from tmem.qa import GENERAL_QA, SESSIONS, score_probe_answer, score_qa_answer


class TestKeywordScore:
    def test_full_and_partial_hits(self):
        assert keyword_score("The KV cache stores Key and Value tensors", ["key", "value"]) == 1.0
        assert keyword_score("only Key here", ["key", "value"]) == 0.5

    def test_case_insensitive_substring(self):
        assert keyword_score("BARSIK is a cat", ["barsik"]) == 1.0

    def test_no_expectation_or_no_hit(self):
        assert keyword_score("anything", []) == 0.0
        assert keyword_score("anything", ["zzz"]) == 0.0


class TestGeneralQA:
    def test_structure(self):
        assert len(GENERAL_QA) >= 5
        for item in GENERAL_QA:
            assert item.expected and all(kw in item.answer.lower() for kw in item.expected), item.question

    def test_reference_answers_score_perfect(self):
        for item in GENERAL_QA:
            assert score_qa_answer(item.answer, item) == 1.0


class TestSessions:
    def test_structure(self):
        for session in SESSIONS:
            assert session.opening and session.probes
            for probe in session.probes:
                assert probe.expected
                assert probe.after_turn >= 1

    def test_facts_are_keyword_findable(self):
        """The planted fact must contain the probe's expected keywords —
        otherwise a perfect memory could still score zero."""
        for session in SESSIONS:
            facts = " ".join(session.opening).lower()
            for probe in session.probes:
                for kw in probe.expected:
                    assert kw.lower() in facts, f"{session.name}: {kw!r} missing from openings"

    def test_probe_answer_scoring(self):
        probe = SESSIONS[0].probes[1]
        assert score_probe_answer("Your cat Barsik is doing great!", probe) == 1.0
        assert score_probe_answer("I have no idea.", probe) == 0.0


class TestQualityMetrics:
    def test_novelty_bounds(self):
        assert novelty_quality("tiny", ["whatever history"]) == 0.1
        text = "completely novel unique words about transformers attention memory caches"
        q = novelty_quality(text, [])
        assert 0 < q <= 1.0
        # repeated text loses novelty
        assert novelty_quality(text, [text]) < q

    def test_strict_quality(self):
        assert strict_quality("short", []) == 0.05
        long_text = " ".join(f"word{i}" for i in range(60))
        assert 0.0 < strict_quality(long_text, []) <= 1.0
        repeated = " ".join(["same"] * 60)
        assert strict_quality(repeated, [long_text]) < strict_quality(long_text, [])

    def test_summarize(self):
        assert summarize([]) == {"mean": None, "min": None, "max": None}
        s = summarize([1.0, 2.0, 3.0])
        assert s == {"mean": 2.0, "min": 1.0, "max": 3.0}

    def test_empty_words_edge(self):
        assert strict_quality("", []) == 0.05

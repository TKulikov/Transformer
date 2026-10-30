"""Dialogue-level tests: the whole strategy grid on tiny offline models."""

from __future__ import annotations

import pytest

from tmem.dialogue import Strategy, default_strategies, run_session
from tmem.kvcache import KVConfig
from tmem.qa import SESSIONS

SESSION = SESSIONS[0]


@pytest.fixture(scope="module", params=["causal", "encdec"])
def any_bundle(request):
    return request.getfixturevalue("causal_bundle" if request.param == "causal" else "encdec_bundle")


class TestStrategyGrid:
    def test_default_grid_covers_all_modes(self):
        names = [s.name for s in default_strategies()]
        assert names == [
            "no_cache_full",
            "classic_kv_full",
            "sparse_sliding",
            "sparse_topk",
            "window_only",
            "window_hybrid_memory",
        ]
        assert len(set(names)) == len(names)

    @pytest.mark.parametrize("bundle_name", ["causal_bundle", "encdec_bundle"])
    def test_all_default_strategies_run(self, bundle_name, request):
        bundle = request.getfixturevalue(bundle_name)
        for strategy in default_strategies(budget=24, sink=2, recent=8):
            run = run_session(bundle, SESSION, strategy, max_new_tokens=6, seed=0)
            summary = run.summary()
            assert summary["turns"] == len(SESSION.user_turns())
            assert summary["prompt_tokens"] > 0
            assert summary["generated_tokens"] > 0
            assert set(summary["probe_scores"]) == {f"turn{p.after_turn}" for p in SESSION.probes}


class TestComparisons:
    def test_window_uses_fewer_prompt_tokens_than_full(self, causal_bundle):
        full = run_session(causal_bundle, SESSION, Strategy("f", context="full"), max_new_tokens=5, seed=0)
        win = run_session(causal_bundle, SESSION, Strategy("w", context="window", window=2), max_new_tokens=5, seed=0)
        assert win.total_prompt_tokens < full.total_prompt_tokens

    def test_sparse_cache_is_bounded(self, causal_bundle):
        run = run_session(
            causal_bundle,
            SESSION,
            Strategy("s", context="full", kv=KVConfig.sparse(20, "sliding", sink=2, recent=8)),
            max_new_tokens=10,
            seed=0,
        )
        assert run.summary()["max_cache_len"] <= 20

    def test_hybrid_memory_layer_accumulates(self, any_bundle):
        bundle = any_bundle
        run = run_session(
            bundle,
            SESSION,
            Strategy("m", context="window", window=2, memory=True, memory_window_crit=0.5, kv=KVConfig.classic()),
            max_new_tokens=5,
            seed=0,
        )
        assert run.corr is not None
        assert run.corr.memory.count_element > 0  # memory actually ingested dialogue states

    def test_no_memory_strategy_has_no_corr(self, causal_bundle):
        run = run_session(causal_bundle, SESSION, Strategy("p", context="full", memory=False), max_new_tokens=3, seed=0)
        assert run.corr is None

    def test_probe_scoring_range(self, causal_bundle):
        run = run_session(causal_bundle, SESSION, Strategy("c", context="full"), max_new_tokens=6, seed=0)
        for turn in run.turns:
            if turn.probe_score is not None:
                assert 0.0 <= turn.probe_score <= 1.0


class TestPlumbing:
    def test_answers_recorded(self, causal_bundle):
        run = run_session(causal_bundle, SESSION, Strategy("a"), max_new_tokens=4, seed=0)
        assert all(isinstance(t.answer_text, str) for t in run.turns)
        assert run.turns[0].user_text == SESSION.user_turns()[0]

    def test_probe_turns_replace_planned_turn(self, causal_bundle):
        run = run_session(causal_bundle, SESSION, Strategy("a"), max_new_tokens=4, seed=0)
        probe_turns = [t for t in run.turns if t.probe is not None]
        assert {t.probe.question for t in probe_turns} == {p.question for p in SESSION.probes}
        assert len(probe_turns) == len(SESSION.probes)

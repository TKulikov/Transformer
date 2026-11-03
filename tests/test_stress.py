"""Tests for the stress battery builders and analysis (offline, tiny models)."""

from __future__ import annotations

import pytest

from tmem.dialogue import Strategy, run_session
from tmem.kvcache import KVConfig
from tmem.stress import (
    analyze_session,
    build_adversarial_session,
    build_long_session,
    stress_sessions,
    stress_strategies,
)


class TestBuilders:
    def test_long_session_structure(self):
        session = build_long_session(n_facts=8)
        turns = session.user_turns()
        assert len(turns) >= 30 and len(session.probes) == 8
        for probe in session.probes:
            assert probe.plant_turn is not None and probe.plant_turn < probe.after_turn
            assert probe.expected
        zones = {"early": 0, "middle": 0, "late": 0}
        total = len(turns)
        for probe in session.probes:
            pos = probe.plant_turn / total
            zones["early" if pos < 1 / 3 else "middle" if pos < 2 / 3 else "late"] += 1
        assert zones["early"] >= 1 and zones["middle"] >= 1  # facts spread across zones

    def test_updates_supersede_values(self):
        session = build_long_session()
        updated = [p for p in session.probes if p.stale]
        assert updated, "at least one updated fact must exist"
        for p in updated:
            assert set(p.expected) & set(p.stale) == set()  # no keyword collision

    def test_adversarial_twins(self):
        session = build_adversarial_session()
        assert len(session.probes) == 6
        twin = next(p for p in session.probes if "first" in p.question)
        assert twin.stale == ["oxidus"]


class TestAnalysis:
    @pytest.fixture(scope="class")
    def mini_analysis(self, causal_bundle):
        session = build_long_session(n_facts=4, paraphrase_every=3)
        strategies = [
            Strategy("classic_kv_full", context="full"),
            Strategy("window2_only", context="window", window=2),
            Strategy("window2_mem_def", context="window", window=2, memory=True, memory_window_crit=0.5),
        ]
        runs = {s.name: run_session(causal_bundle, session, s, max_new_tokens=3, seed=0) for s in strategies}
        return analyze_session(runs)

    def test_all_metrics_present(self, mini_analysis):
        for name, a in mini_analysis.items():
            for key in ("recall_mean", "recall_direct", "recall_paraphrase", "recall_by_zone",
                        "recall_by_lag", "stale_rate", "consistency_vs_reference", "prompt_tokens",
                        "generated_tokens", "cache_slots", "median_turn_latency_s", "p90_turn_latency_s",
                        "peak_mem_mb", "memory_stats"):
                assert key in a, (name, key)
            assert 0.0 <= a["recall_mean"] <= 1.0
            assert set(a["recall_by_lag"]) == {"0-5", "6-12", "13-20", "21+"}
            assert all(value is None or 0.0 <= value <= 1.0 for value in a["recall_by_lag"].values())

    def test_reference_is_perfectly_consistent_with_itself(self, mini_analysis):
        assert mini_analysis["classic_kv_full"]["consistency_vs_reference"] == 1.0

    def test_memory_strategy_reports_stats(self, mini_analysis):
        stats = mini_analysis["window2_mem_def"]["memory_stats"]
        assert stats and stats["get_calls"] > 0 and stats["update_calls"] > 0

    def test_grid_size(self):
        grid = stress_strategies()
        assert len(grid) == 11 and len({s.name for s in grid}) == 11
        assert len(stress_sessions()) == 2

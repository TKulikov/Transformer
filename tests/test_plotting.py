"""Offline smoke tests for the plotting report generation."""

from __future__ import annotations

import pytest

from tmem.dialogue import Strategy, run_session
from tmem.plotting import plot_cache_growth, plot_context_growth, plot_efficiency_quality
from tmem.qa import SESSIONS


@pytest.fixture(scope="module")
def tiny_runs(causal_bundle):
    session = SESSIONS[0]
    short = Strategy("classic", context="full")
    win = Strategy("window", context="window", window=2)
    mem = Strategy("hybrid", context="window", window=2, memory=True, memory_window_crit=0.5)
    return {
        s.name: run_session(causal_bundle, session, s, max_new_tokens=3, seed=0)
        for s in (short, win, mem)
    }


def test_all_plots_render(tiny_runs, tmp_path):
    for fn in (plot_context_growth, plot_cache_growth, plot_efficiency_quality):
        path = fn(tiny_runs, str(tmp_path / f"{fn.__name__}.png"))
        import os

        assert os.path.getsize(path) > 1000  # a real PNG was written

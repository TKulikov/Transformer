"""Tests for the corpus statistics kernel and INTERIM rendering (offline)."""

from __future__ import annotations

import json

import pytest

from tmem.corpus_stats import bootstrap_ci, compare_pairs, holm, load_run, paired_bootstrap_p, render_interim_md, summarize


class TestBootstrap:
    def test_ci_contains_mean_and_narrows(self):
        wide = bootstrap_ci([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        narrow = bootstrap_ci([0.5] * 20 + [0.51, 0.49])
        assert wide[0] <= 0.5 <= wide[1]
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_ci_single_value(self):
        assert bootstrap_ci([0.42]) == (0.42, 0.42)

    def test_ci_empty(self):
        assert bootstrap_ci([]) == (None, None)

    def test_paired_p_identical_is_one(self):
        vals = [0.5, 0.6, 0.7, 0.4]
        assert paired_bootstrap_p(vals, vals) == 1.0

    def test_paired_p_detects_shift(self):
        a = [0.9, 0.85, 0.92, 0.88, 0.91, 0.87]
        b = [0.4, 0.45, 0.42, 0.38, 0.41, 0.44]
        assert paired_bootstrap_p(a, b) < 0.05

    def test_holm_monotone_and_capped(self):
        adj = holm({"a": 0.01, "b": 0.04, "c": 0.6})
        assert adj["a"] <= adj["b"] <= adj["c"] <= 1.0
        assert adj["a"] == pytest.approx(0.03)


# ------------------------------------------------------------ run fixtures

ANALYSIS_TEMPLATE = {
    "recall_mean": 0.5, "recall_direct": 0.6, "recall_paraphrase": 0.4,
    "recall_by_kind": {"direct": 0.6, "paraphrase": 0.4, "multihop": 0.5,
                        "temporal": None, "unanswerable": 1.0},
    "recall_by_lag": {"0-5": 0.7, "6-12": 0.5, "13-20": 0.4, "21+": 0.3},
    "stale_rate": 0.1, "tone_mean": 0.8, "prompt_tokens": 1000, "cache_slots": 5000,
}


def _make_run(tmp_path, n_batches=4, strategies=("v4_rag_kv", "rag_text_only")):
    run_dir = tmp_path / "run"
    specs = [
        {"spec_id": f"b{i:02d}", "topic": ["work", "travel"][i % 2], "persona": "terse",
         "scenario": "baseline", "target_turns": 100, "seed": i}
        for i in range(n_batches)
    ]
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_id": "test-run", "preset": "pilot", "model": "fake", "git_sha": "abc1234",
        "specs": specs, "strategies": list(strategies),
        "protocol": {"latency_repeats": 5},
    }))
    with (run_dir / "manifest.jsonl").open("w") as fh:
        for i, spec in enumerate(specs):
            batch_dir = run_dir / "batches" / spec["spec_id"]
            batch_dir.mkdir(parents=True)
            for j, sname in enumerate(strategies):
                analysis = dict(ANALYSIS_TEMPLATE)
                analysis["recall_mean"] = 0.8 - 0.3 * j + 0.01 * i  # v4 better than rag
                (batch_dir / f"{sname}.json").write_text(json.dumps({
                    "strategy": sname, "spec_id": spec["spec_id"], "analysis": analysis,
                    "e2e": {"per_rep_mean_s": [1.0 + 0.1 * j] * 5, "mean_s": 1.0 + 0.1 * j,
                             "median_s": 1.0 + 0.1 * j, "includes_retrieval": True},
                }))
            fh.write(json.dumps({"batch_id": spec["spec_id"], "status": "done",
                                  "wall_s": 60.0, "reruns": 0, "spec": spec}) + "\n")
    return run_dir


class TestRunSummary:
    def test_load_and_summarize(self, tmp_path):
        run_dir = _make_run(tmp_path)
        run = load_run(run_dir)
        assert len(run["records"]) == 4
        summary = summarize(run)
        assert summary["n_dialogues_done"] == 4 and summary["n_dialogues_total"] == 4
        v4 = summary["strategies"]["v4_rag_kv"]
        assert v4["recall_mean"] > summary["strategies"]["rag_text_only"]["recall_mean"]
        lo, hi = v4["recall_ci95"]
        assert lo <= v4["recall_mean"] <= hi
        assert v4["e2e_latency_median_s"] is not None
        assert set(v4["slices"]["topic"]) == {"travel", "work"}

    def test_load_dedups_reruns(self, tmp_path):
        run_dir = _make_run(tmp_path)
        with (run_dir / "manifest.jsonl").open("a") as fh:  # rerun of b00
            fh.write(json.dumps({"batch_id": "b00", "status": "done", "wall_s": 70.0,
                                  "reruns": 1, "spec": {"spec_id": "b00"}}) + "\n")
        run = load_run(run_dir)
        assert len(run["records"]) == 4

    def test_compare_pairs_and_render(self, tmp_path):
        run_dir = _make_run(tmp_path, n_batches=6)
        run = load_run(run_dir)
        pvals = compare_pairs(run, [("v4_rag_kv", "rag_text_only")])
        assert "v4_rag_kv vs rag_text_only" in pvals
        md = render_interim_md(run_dir)
        assert "6/6" in md and "v4_rag_kv" in md and "Прогресс" in md
        assert "Recall по lag-зонам" in md and "Попарные сравнения" in md

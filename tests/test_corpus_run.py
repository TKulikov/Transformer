"""Tests for the batch runner engine: resume, atomicity, dry-run (offline, stub runner)."""

from __future__ import annotations

import json

import pytest

from tmem.corpus import CorpusSpec
from tmem.corpus_run import load_manifest, run_corpus
from tmem.dialogue import Strategy

STRATEGIES = [Strategy("fake_a"), Strategy("fake_b")]

ANALYSIS = {
    "recall_mean": 0.75, "recall_direct": 0.8, "recall_paraphrase": 0.7,
    "recall_by_kind": {"direct": 0.8, "paraphrase": 0.7, "multihop": 0.6,
                        "temporal": None, "unanswerable": 1.0},
    "recall_by_lag": {"0-5": 0.9, "6-12": 0.8, "13-20": 0.7, "21+": 0.6},
    "stale_rate": 0.1, "tone_mean": 0.8, "prompt_tokens": 1000, "cache_slots": 5000,
}

META = {"run_id": "t-run", "preset": "pilot", "model": "stub", "git_sha": "deadbeef",
        "protocol": {"latency_repeats": 5}}


def _specs(n=3):
    return [
        CorpusSpec(spec_id=f"b{i:02d}", topic="work", persona="terse", scenario="baseline",
                   target_turns=100, seed=i)
        for i in range(n)
    ]


def _stub_runner(log=None, fail_on=(), raise_on=()):
    """Runner stub producing valid batch payloads without any model."""

    def runner(spec):
        if log is not None:
            log.append(spec.spec_id)
        if spec.spec_id in raise_on:
            raise RuntimeError("simulated crash")
        if spec.spec_id in fail_on:
            return {"errors": ["simulated invalid session"], "results": {}}
        results = {
            s.name: {"analysis": dict(ANALYSIS),
                      "e2e": {"per_rep_mean_s": [1.0] * 5, "mean_s": 1.0, "median_s": 1.0,
                               "includes_retrieval": True}}
            for s in STRATEGIES
        }
        return {"results": results, "errors": [], "n_probes": 20, "n_turns": 100}

    return runner


class TestRunCorpus:
    def test_full_run_writes_outputs(self, tmp_path):
        run_dir = tmp_path / "run"
        out = run_corpus(run_dir, specs=_specs(), strategies=STRATEGIES,
                         runner=_stub_runner(), meta=META)
        assert out["done"] == 3
        manifest = load_manifest(run_dir)
        assert {k: v["status"] for k, v in manifest.items()} == {"b00": "done", "b01": "done", "b02": "done"}
        for b in ("b00", "b01", "b02"):
            files = sorted(p.name for p in (run_dir / "batches" / b).glob("*.json"))
            assert files == ["fake_a.json", "fake_b.json"]
        assert (run_dir / "run_config.json").exists()
        interim = (run_dir / "INTERIM.md").read_text()
        assert "3/3" in interim and "fake_a" in interim

    def test_batches_limit_and_resume(self, tmp_path):
        run_dir = tmp_path / "run"
        log = []
        run_corpus(run_dir, specs=_specs(3), strategies=STRATEGIES,
                   runner=_stub_runner(log), meta=META, batches=2)
        assert log == ["b00", "b01"]
        run_corpus(run_dir, specs=_specs(3), strategies=STRATEGIES,
                   runner=_stub_runner(log), meta=META)
        assert log == ["b00", "b01", "b02"], "resume must run only the pending batch"

    def test_crashed_batch_is_marked_failed_and_retried(self, tmp_path):
        run_dir = tmp_path / "run"
        run_corpus(run_dir, specs=_specs(2), strategies=STRATEGIES,
                   runner=_stub_runner(raise_on={"b01"}), meta=META)
        manifest = load_manifest(run_dir)
        assert manifest["b00"]["status"] == "done"
        assert manifest["b01"]["status"] == "failed"
        assert not (run_dir / "batches" / "b01").exists()
        log = []
        run_corpus(run_dir, specs=_specs(2), strategies=STRATEGIES,
                   runner=_stub_runner(log), meta=META)
        assert log == ["b01"]
        manifest = load_manifest(run_dir)
        assert manifest["b01"]["status"] == "done" and manifest["b01"]["reruns"] == 1

    def test_validation_failure_recorded(self, tmp_path):
        run_dir = tmp_path / "run"
        run_corpus(run_dir, specs=_specs(1), strategies=STRATEGIES,
                   runner=_stub_runner(fail_on={"b00"}), meta=META)
        manifest = load_manifest(run_dir)
        assert manifest["b00"]["status"] == "failed"
        assert manifest["b00"]["errors"]

    def test_dry_run_writes_nothing(self, tmp_path, capsys):
        run_dir = tmp_path / "dry"
        log = []
        out = run_corpus(run_dir, specs=_specs(2), strategies=STRATEGIES,
                         runner=_stub_runner(log), meta=META, dry_run=True)
        assert log == [] and out["done"] == 0
        assert not run_dir.exists()
        assert "b00" in capsys.readouterr().out

    def test_only_reruns_finished_batch(self, tmp_path):
        run_dir = tmp_path / "run"
        run_corpus(run_dir, specs=_specs(2), strategies=STRATEGIES,
                   runner=_stub_runner(), meta=META)
        log = []
        run_corpus(run_dir, specs=_specs(2), strategies=STRATEGIES,
                   runner=_stub_runner(log), meta=META, only=["b00"])
        assert log == ["b00"]
        assert load_manifest(run_dir)["b00"]["reruns"] == 1

    def test_queue_is_frozen_by_run_config(self, tmp_path):
        run_dir = tmp_path / "run"
        run_corpus(run_dir, specs=_specs(2), strategies=STRATEGIES,
                   runner=_stub_runner(), meta=META)
        extended = _specs(2) + [CorpusSpec(spec_id="new-one", topic="travel", persona="terse",
                                            scenario="baseline", target_turns=100, seed=9)]
        with pytest.raises(SystemExit):
            run_corpus(run_dir, specs=extended, strategies=STRATEGIES,
                       runner=_stub_runner(), meta=META)

    def test_partial_run_interim_shows_progress(self, tmp_path):
        run_dir = tmp_path / "run"
        run_corpus(run_dir, specs=_specs(4), strategies=STRATEGIES,
                   runner=_stub_runner(), meta=META, batches=2)
        interim = (run_dir / "INTERIM.md").read_text()
        assert "2/4" in interim
        assert json.loads((run_dir / "run_config.json").read_text())["run_id"] == "t-run"

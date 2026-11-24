"""End-to-end smoke: one tiny corpus batch through ModelBatchRunner on CPU.

Uses the session-scoped ``causal_bundle`` fixture (tiny random model), so no
GPU and no downloads are needed. Verifies that the new probe kinds survive
``run_session`` and that ``analyze_session`` output feeds the runner contract.
"""

from __future__ import annotations

import json

from tmem.corpus import CorpusSpec
from tmem.corpus_run import ModelBatchRunner, run_corpus
from tmem.dialogue import Strategy

TINY_SPEC = CorpusSpec(
    spec_id="smoke-00-work-terse-updates", topic="work", persona="terse", scenario="updates",
    target_turns=24, seed=11, n_facts=4, probes_per_zone=1, n_unanswerable=1, n_temporal=1,
)

TINY_STRATEGIES = [
    Strategy("classic_kv_full", context="full"),
    Strategy("window2_only", context="window", window=2),
]


def test_tiny_batch_end_to_end(causal_bundle, tmp_path):
    runner = ModelBatchRunner(causal_bundle, TINY_STRATEGIES, max_new_tokens=2,
                              seed=0, latency_repeats=1)
    meta = {"run_id": "smoke", "preset": "smoke", "model": "tiny-causal",
            "git_sha": "test", "protocol": {"latency_repeats": 1}}
    out = run_corpus(tmp_path / "smoke", specs=[TINY_SPEC], strategies=TINY_STRATEGIES,
                     runner=runner, meta=meta)
    assert out["done"] == 1, "tiny batch must complete"

    manifest = [json.loads(line) for line in (tmp_path / "smoke" / "manifest.jsonl").read_text().splitlines()]
    assert manifest[-1]["status"] == "done"
    assert manifest[-1]["n_probes"] >= 4  # direct/paraphrase/multihop/temporal/unanswerable

    for strategy in TINY_STRATEGIES:
        payload = json.loads((tmp_path / "smoke" / "batches" / TINY_SPEC.spec_id / f"{strategy.name}.json").read_text())
        analysis = payload["analysis"]
        assert 0.0 <= analysis["recall_mean"] <= 1.0
        assert analysis["recall_by_kind"], "kind breakdown must be present"
        e2e = payload["e2e"]
        assert e2e["includes_retrieval"] is True
        assert len(e2e["per_rep_mean_s"]) == 1  # latency_repeats = 1
        assert e2e["median_s"] is not None and e2e["median_s"] > 0

    interim = (tmp_path / "smoke" / "INTERIM.md").read_text()
    assert "1/1" in interim and "classic_kv_full" in interim


def test_rag_variants_end_to_end_keep_kv_isolated():
    from scripts.smoke_latent_campaign import tiny_qwen35_bundle
    from tmem.corpus_run import full_rag_strategies

    bundle = tiny_qwen35_bundle()
    selected = {
        "rag_mid_centered", "rag_mean_centered", "rag_bank_dev",
        "v4_rag_kv_mid", "v4_rag_kv_mean_centered",
    }
    strategies = [s for s in full_rag_strategies() if s.name in selected]
    runner = ModelBatchRunner(bundle, strategies, max_new_tokens=2,
                              seed=0, latency_repeats=0)
    out = runner(TINY_SPEC)

    assert out["errors"] == []
    for name in ("rag_mid_centered", "rag_mean_centered", "rag_bank_dev"):
        analysis = out["results"][name]["analysis"]
        assert analysis["kv_steps"] is None
        assert analysis["rag_stats"]["size"] > 0
    for name in ("v4_rag_kv_mid", "v4_rag_kv_mean_centered"):
        dual = out["results"][name]["analysis"]
        assert dual["kv_steps"] > 0
        assert dual["kv_injected_slots"] > 0

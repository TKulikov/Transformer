"""CPU smoke of the latent campaign strategy set (no GPU, tiny models).

Validates that all 10 strategies of `--strategy-set latent` run end-to-end
through ModelBatchRunner and produce the full analysis payload:
  * non-injection strategies on the tiny GPT-2 bundle;
  * v4_* strategies (K/V injection) on a 4-layer full-attention Qwen3.5
    with random weights (same builder as scripts/verify_paper.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from tmem.corpus import CorpusSpec  # noqa: E402
from tmem.corpus_run import ModelBatchRunner, latent_strategies  # noqa: E402
from tmem.models import ModelBundle  # noqa: E402

TINY_SPEC = CorpusSpec(
    spec_id="smoke-latent-work-terse-updates", topic="work", persona="terse", scenario="updates",
    target_turns=24, seed=11, n_facts=4, probes_per_zone=1, n_unanswerable=1, n_temporal=1,
)

META = {"run_id": "smoke-latent", "preset": "smoke", "model": "tiny", "git_sha": "smoke",
        "protocol": {"latency_repeats": 1}}


def tiny_qwen35_bundle() -> ModelBundle:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from tmem.models import ByteTokenizer

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=256, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128,
        layer_types=["full_attention"] * 4,
    )
    model = M.Qwen3_5ForCausalLM(cfg).eval()
    return ModelBundle(name="tiny-qwen35", kind="causal", model=model,
                       tokenizer=ByteTokenizer(), device=torch.device("cpu"))


def check_payload(strategy: str, payload: dict) -> None:
    a = payload["analysis"]
    for key in ("recall_mean", "recall_by_kind", "recall_by_lag", "stale_rate",
                "tone_mean", "rep_mean", "prompt_tokens", "probe_details"):
        assert key in a, (strategy, key)
    assert 0.0 <= a["recall_mean"] <= 1.0
    assert a["probe_details"], strategy
    assert payload["e2e"]["includes_retrieval"] is True
    assert a["recall_by_kind"].get("unanswerable") is not None


def main() -> int:
    from tmem.models import tiny_causal

    all_names = [s.name for s in latent_strategies()]
    print("strategy set:", all_names)
    assert len(all_names) == len(set(all_names)) == 10

    # non-injection strategies on tiny GPT-2 (snapkv/pyramidkv need per-layer
    # masks GPT-2 lacks — same restriction as test_battery_qwen35)
    causal_ok = {"classic_kv_full", "sparse_sliding_128", "sparse_topk_128",
                 "rag_text_only", "rag_latent_mid"}
    bundle = tiny_causal()
    strategies = [s for s in latent_strategies() if s.name in causal_ok]
    out = ModelBatchRunner(bundle, strategies, max_new_tokens=2, seed=0,
                           latency_repeats=1)(TINY_SPEC)
    assert not out["errors"], out["errors"]
    for s in strategies:
        check_payload(s.name, out["results"][s.name])
        print(f"  OK causal {s.name}: recall={out['results'][s.name]['analysis']['recall_mean']}")

    # injection strategies on tiny Qwen3.5
    qbundle = tiny_qwen35_bundle()
    kv_strategies = [s for s in latent_strategies()
                     if s.name in {"v4_rag_kv", "v4_rag_kv_mid", "v4_kv_mid"}]
    out2 = ModelBatchRunner(qbundle, kv_strategies, max_new_tokens=2, seed=0,
                            latency_repeats=1)(TINY_SPEC)
    assert not out2["errors"], out2["errors"]
    for s in kv_strategies:
        check_payload(s.name, out2["results"][s.name])
        a = out2["results"][s.name]["analysis"]
        kv = a.get("kv_injected_slots") or 0
        print(f"  OK qwen35 {s.name}: recall={a['recall_mean']} kv_injected={kv}")
        assert kv > 0, f"{s.name}: K/V injection did not fire on tiny qwen35"

    print("SMOKE LATENT: all 10 strategies OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

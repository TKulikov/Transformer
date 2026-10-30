"""Integration tests on *real* checkpoints (needs network + HF cache).

Run explicitly:  pytest -m integration -q
These validate the whole pipeline against genuine tokenizers and weights,
still on CPU — the GPU runs are a separate, approved step.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.integration

from tmem.dialogue import Strategy, run_session  # noqa: E402
from tmem.kvcache import KVConfig  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.qa import SESSIONS  # noqa: E402


@pytest.fixture(scope="module")
def gpt2_bundle():
    return load_model("gpt2", device="cpu", dtype=torch.float32, eager=True)


def test_gpt2_classic_cache_matches_recompute(gpt2_bundle):
    ids = gpt2_bundle.tokenizer("The capital of France is", return_tensors="pt")["input_ids"]
    from tmem.generate import decode_causal

    a = decode_causal(gpt2_bundle.model, ids, cache="none", max_new_tokens=8)
    b = decode_causal(gpt2_bundle.model, ids, kv=KVConfig.classic(), cache="use", max_new_tokens=8)
    assert a.token_ids == b.token_ids
    assert gpt2_bundle.tokenizer.decode(b.token_ids).strip() != ""


def test_gpt2_sparse_cache_bounded_and_runs(gpt2_bundle):
    from tmem.generate import decode_causal

    ids = gpt2_bundle.tokenizer("Explain the transformer architecture in one paragraph.", return_tensors="pt")["input_ids"]
    res = decode_causal(gpt2_bundle.model, ids, kv=KVConfig.sparse(48, "topk", sink=4, recent=24), cache="use", max_new_tokens=16)
    assert max(res.cache_lengths) <= 48
    assert res.steps == 16


def test_gpt2_dialogue_strategies(gpt2_bundle):
    classic = run_session(gpt2_bundle, SESSIONS[0], Strategy("classic", context="full"), max_new_tokens=12, seed=0)
    sparse = run_session(
        gpt2_bundle,
        SESSIONS[0],
        Strategy("sparse", context="full", kv=KVConfig.sparse(64, "sliding", sink=4, recent=16)),
        max_new_tokens=12,
        seed=0,
    )
    hybrid = run_session(
        gpt2_bundle, SESSIONS[0], Strategy("hybrid", context="window", window=3, memory=True), max_new_tokens=12, seed=0
    )
    assert sparse.summary()["max_cache_len"] <= 64
    assert classic.summary()["max_cache_len"] >= sparse.summary()["max_cache_len"]
    assert hybrid.corr.memory is not None
    for run in (classic, sparse, hybrid):
        assert run.turns and all(isinstance(t.answer_text, str) for t in run.turns)

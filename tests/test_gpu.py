"""GPU-only checks. Excluded from the default run.

Execute deliberately (after approval) with a ROCm/CUDA build of torch:
    pytest -m gpu -q
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator (CUDA/ROCm build of torch)"),
]


@pytest.fixture(scope="module")
def gpt2_gpu():
    from tmem.models import load_model

    return load_model("gpt2", device="cuda", dtype=torch.float32, eager=True)


def test_memory_lives_on_gpu(gpt2_gpu):
    from tmem.memory import CorrectionLayer

    corr = CorrectionLayer(gpt2_gpu.hidden_size)
    x = torch.randn(1, 8, gpt2_gpu.hidden_size, device="cuda")
    corr(x)  # first use migrates tensors to the query device
    assert corr.memory.context.device.type == "cuda"
    assert corr.memory.get_relevant(x, 8)[1] >= 0


def test_classic_vs_sparse_latency_gpu(gpt2_gpu):
    from tmem.generate import decode_causal
    from tmem.kvcache import KVConfig

    long_prompt = gpt2_gpu.tokenizer(("Context: " + "transformers memory caches " * 40), return_tensors="pt")["input_ids"]
    classic = decode_causal(gpt2_gpu.model, long_prompt.cuda(), kv=KVConfig.classic(), cache="use", max_new_tokens=24)
    sparse = decode_causal(gpt2_gpu.model, long_prompt.cuda(), kv=KVConfig.sparse(128, "topk", sink=4, recent=48), cache="use", max_new_tokens=24)
    assert max(sparse.cache_lengths) <= 128 < max(classic.cache_lengths)
    assert sparse.total_cache_slots < classic.total_cache_slots

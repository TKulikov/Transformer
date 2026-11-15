"""Phase-1 spike: verify model load + hidden-state embeddings on the hybrid.

Checks VRAM is free, loads Qwen3.5-4B, extracts embeddings at several mid
full-attention layers, sanity-checks cosine (self=1, distinct<1), and runs one
short generation. Run on the GPU:

    HF_HUB_OFFLINE=1 .venv/bin/python scripts_code/spike_embed.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from codemem.model import load_model
from codemem.vector import cosine_sim


def main() -> None:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM free {free/1024**3:.1f} / {total/1024**3:.1f} GiB")
        if free / 1024**3 < 10:
            raise SystemExit("VRAM < 10 GiB free — aborting (don't share the GPU)")
    t0 = time.time()
    cm = load_model(device="cuda", dtype="bfloat16")
    print(f"loaded in {time.time()-t0:.1f}s; hidden_size={cm.hidden_size}")

    texts = [
        "def hash_password(pw): return hashlib.sha256(pw.encode()).hexdigest()",
        "def login(user, pw): check credentials against the database",
        "the quick brown fox jumps over the lazy dog",
    ]
    for layer in (11, 15, 19, 23):
        emb = cm.embed(texts, layer=layer)
        self_sim = float(cosine_sim(emb[0], emb[0:1])[0])
        cross = float(cosine_sim(emb[0], emb[1:2])[0])
        unrelated = float(cosine_sim(emb[0], emb[2:3])[0])
        print(f"layer {layer:2d}: shape {tuple(emb.shape)} self={self_sim:.4f} "
              f"code-code={cross:.4f} code-text={unrelated:.4f}")

    t0 = time.time()
    out = cm.generate("Write a Python function that adds two numbers.", max_new_tokens=64)
    print(f"generate ({time.time()-t0:.1f}s): {out[:200]!r}")


if __name__ == "__main__":
    main()
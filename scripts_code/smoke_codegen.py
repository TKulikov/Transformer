"""Phase-3 smoke: end-to-end goal-conditioned chunked code generation.

Exercises the full wiring on one small request: goal build (generate/compress/
fidelity) -> subgoal decomposition (dendrogram + judge) -> per-chunk plan ->
prefetch prediction -> unified retrieval -> generation -> write-back -> use
scores. Saves a summary to results/code/smoke/. Run on the GPU:

    HF_HUB_OFFLINE=1 .venv/bin/python scripts_code/smoke_codegen.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from codemem.bank import CodeBank
from codemem.config import Config
from codemem.model import load_model
from codemem.planner import CodePlanner
from codemem.prefetch import PrefetchBank
from codemem.retrieve import Retriever
from codemem.tree import GoalTree


def main() -> None:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM free {free/1024**3:.1f} / {total/1024**3:.1f} GiB", flush=True)
        if free / 1024**3 < 10:
            raise SystemExit("VRAM < 10 GiB free — aborting")
    cfg = Config(chunk_max_tokens=160, plan_max_tokens=64, goal_max_tokens=96, compress_max_tokens=32)
    t0 = time.time()
    model = load_model(device="cuda", dtype=cfg.dtype, embed_layer=cfg.embed_layer)
    print(f"model loaded {time.time()-t0:.1f}s", flush=True)

    bank = CodeBank(use_boost=cfg.use_boost, verified_boost=cfg.verified_boost)
    tree = GoalTree(bank, tau_assimilate=cfg.tau_assimilate, centered=cfg.centered, mmr_lambda=cfg.mmr_lambda)
    prefetch = PrefetchBank(ttl_steps=cfg.prefetch_ttl_steps)
    retriever = Retriever(tree, bank, prefetch, top_k=cfg.top_k, mmr_lambda=cfg.mmr_lambda, centered=cfg.centered)
    planner = CodePlanner(model, cfg, bank, tree, prefetch, retriever)

    request = ("Write a small Python module for password hashing and login "
               "verification using a dict as the user database.")
    t0 = time.time()
    goal, chunks = planner.run(request, n_chunks=2)
    dt = time.time() - t0

    print(f"\nGOAL: {goal.compressed!r}  fidelity={goal.fidelity:.3f}", flush=True)
    print(f"full goal: {goal.text!r}", flush=True)
    print(f"tree nodes={len(tree.nodes)} bank chunks={len(bank)} generated={len(chunks)} in {dt:.1f}s", flush=True)
    for i, c in enumerate(chunks):
        print(f"\n--- chunk {i} ---\n{c[:400]}", flush=True)

    out = Path("results/code/smoke")
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "request": request,
        "goal_full": goal.text,
        "goal_compressed": goal.compressed,
        "fidelity": goal.fidelity,
        "tree_nodes": len(tree.nodes),
        "bank_chunks": len(bank),
        "chunks": chunks,
        "seconds": round(dt, 1),
        "nodes": [
            {"id": n.id, "text": n.text[:80], "depth": n.depth, "parent": n.parent,
             "solved": n.solved, "chunks": n.chunk_ids}
            for n in tree.nodes.values()
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved {out/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
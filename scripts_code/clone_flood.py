"""Clone-flooding diagnostic: can MMR diversification surface utils.retry?

In the r2 ablation utils.retry was missed by ALL conditions — buried under
near-duplicate backoff_delay/clamp chunks in the utils grab-bag. This sweeps the
MMR lambda over the utils needs (goal-routed, uncentered — the best r2 config)
and reports the target rank per lambda, focusing on retry. Saves
results/code/clone_flood/<run-id>/. GPU run:

    HF_HUB_OFFLINE=1 .venv/bin/python scripts_code/clone_flood.py --run-id cf1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from codemem.bank import CodeBank
from codemem.config import Config
from codemem.model import load_model
from codemem.prefetch import PrefetchBank
from codemem.retrieve import Retriever
from eval_retrieval import GOALS, TASKS, build_goal, extract_functions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="cf1")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM free {free/1024**3:.1f} / {total/1024**3:.1f} GiB", flush=True)
        if free / 1024**3 < 10:
            raise SystemExit("VRAM < 10 GiB free — aborting")
    cfg = Config(embed_layer=args.layer)
    model = load_model(device="cuda", dtype=cfg.dtype, embed_layer=args.layer)

    funcs = extract_functions()
    embs = model.embed([src for _, _, src in funcs])
    bank = CodeBank()
    chunks = [(bank.add(src, emb), mod, q) for (q, mod, src), emb in zip(funcs, embs)]
    qname_to_id = {q: cid for cid, _, q in chunks}
    goal_embs = {g: model.embed([d])[0] for g, d in GOALS.items()}
    tree = build_goal(bank, chunks, goal_embs, centered=False)   # goal_uncentered (best)

    utils_tasks = [(need, qname_to_id[target], target) for g, need, target in TASKS if g == "utils"]
    need_embs = model.embed([n for n, _, _ in utils_tasks])
    lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]

    table = {}
    for lam in lambdas:
        retr = Retriever(tree, bank, PrefetchBank(), top_k=args.k, mmr_lambda=lam, centered=False)
        ranks = {}
        for i, (need, tid, target) in enumerate(utils_tasks):
            out = retr.retrieve(goal_embs["utils"], need_embs[i], i, k=args.k)
            ranks[target] = out.index(tid) + 1 if tid in out else None
        hit6 = sum(1 for r in ranks.values() if r is not None and r <= 6) / len(ranks)
        table[lam] = {"ranks": ranks, "hit@6": round(hit6, 3)}
        print(f"lambda={lam}: hit@6={hit6:.3f} retry={ranks['utils.retry']}", flush=True)

    out = Path(f"results/code/clone_flood/{args.run_id}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"layer": args.layer, "k": args.k, "table": table}, indent=2))
    targets = [t for _, _, t in utils_tasks]
    lines = ["# Clone-flooding: MMR lambda sweep (utils, goal-routed uncentered)", "",
             f"layer={args.layer} k={args.k}", "",
             "| target | " + " | ".join(f"λ={l}" for l in lambdas) + " |",
             "|---|" + "---|" * len(lambdas)]
    for t in targets:
        lines.append(f"| {t} | " + " | ".join(str(table[l]['ranks'][t]) for l in lambdas) + " |")
    lines.append("| **hit@6** | " + " | ".join(str(table[l]['hit@6']) for l in lambdas) + " |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
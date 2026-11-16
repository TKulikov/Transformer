"""Confidence-adaptive routing: can practical goal-routing beat flat?

s2 showed centroid routing lifts generated-goal hit@6 to 0.667 (from 0.267) but
still <= flat (0.733); a bad goal estimate can hurt worse than flat. This tests
confidence-adaptive routing: goal-routing when the goal+need blend is confident
(top cosine to node centroids >= threshold), else FLAT fallback. Guarantees
flat-level performance on uncertain goals and captures goal gain on confident ones.

Conditions: flat_baseline, gen_soft_centroid (s2 best), adaptive at thresholds
0.3/0.5/0.7/0.9, oracle_greedy (upper bound). Reports hit@k, MRR, bootstrap CI,
goal-mode fraction and routing accuracy within goal-mode. Saves
results/code/adaptive/<run-id>/. GPU run:

    HF_HUB_OFFLINE=1 .venv/bin/python scripts_code/eval_adaptive.py --run-id a1
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
from codemem.goal import GoalBuilder
from codemem.model import load_model
from codemem.prefetch import PrefetchBank
from codemem.retrieve import Retriever
from codemem.vector import cosine_sim, normalize
from eval_retrieval import (GOALS, TASKS, bootstrap_ci, build_goal, extract_functions,
                            metric_vec, mrr_vec)

# name, method, conf_threshold
CONDS = [
    ("flat_baseline",     "adaptive", 2.0),
    ("gen_soft_centroid", "soft",     None),
    ("adaptive_t0.3",     "adaptive", 0.3),
    ("adaptive_t0.5",     "adaptive", 0.5),
    ("adaptive_t0.7",     "adaptive", 0.7),
    ("adaptive_t0.9",     "adaptive", 0.9),
    ("oracle_greedy",     "oracle",   None),
]


def confidence(tree, route_emb, by_centroid=True):
    children = tree.nodes[tree.root].children
    def kemb(nid):
        node = tree.nodes[nid]
        return node.centroid.value if (by_centroid and node.centroid.value is not None) else node.emb
    return max((float(cosine_sim(route_emb, kemb(c)[None])[0]) for c in children), default=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="a1")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--paraphrases", type=int, default=2)
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
    tree = build_goal(bank, chunks, goal_embs, centered=False)
    goal_node_id = {}
    for nid, n in tree.nodes.items():
        for g, desc in GOALS.items():
            if n.text == desc:
                goal_node_id[g] = nid
    print(f"corpus: {len(funcs)} functions", flush=True)

    gb = GoalBuilder(model, cfg)
    gen_goal_embs = {ti: gb.build(f"I need to {need}").emb for ti, (g, need, _) in enumerate(TASKS)}
    queries = []
    for ti, (g, need, target) in enumerate(TASKS):
        variants = [need]
        for _ in range(args.paraphrases):
            p = model.generate(f"Rewrite this coding need in different words, keep the meaning, one sentence: {need}",
                               max_new_tokens=48)
            variants.append(p.strip() or need)
        for v in variants:
            queries.append((g, v, qname_to_id[target], ti))
    need_embs = model.embed([q[1] for q in queries])
    print(f"queries: {len(queries)}", flush=True)

    retr = Retriever(tree, bank, PrefetchBank(), top_k=args.k, mmr_lambda=0.5, centered=False)
    results, detail = {}, {}
    for name, method, thr in CONDS:
        ranks, goal_mode, goal_correct = [], 0, 0
        for i, (g, need, tid, ti) in enumerate(queries):
            ne = need_embs[i]
            if method == "oracle":
                out = retr.retrieve(goal_embs[g], ne, i, k=args.k)
            elif method == "soft":
                out = retr.retrieve_soft(gen_goal_embs[ti], ne, i, k=args.k, m=3, need_weight=0.5, by_centroid=True)
            else:
                route_emb = normalize(normalize(gen_goal_embs[ti]) + 0.5 * normalize(ne))
                conf = confidence(tree, route_emb)
                if conf >= thr:
                    goal_mode += 1
                    if tree.route_soft(route_emb, m=1, by_centroid=True)[0][0] == goal_node_id[g]:
                        goal_correct += 1
                out = retr.retrieve_adaptive(gen_goal_embs[ti], ne, i, k=args.k, m=3, need_weight=0.5,
                                             conf_threshold=thr, by_centroid=True)
            ranks.append(out.index(tid) + 1 if tid in out else None)
        h6, mrr = metric_vec(ranks, args.k), mrr_vec(ranks)
        results[name] = {
            "hit@1": round(sum(metric_vec(ranks, 1)) / len(ranks), 3),
            "hit@3": round(sum(metric_vec(ranks, 3)) / len(ranks), 3),
            "hit@6": round(sum(h6) / len(ranks), 3), "hit@6_ci": bootstrap_ci(h6),
            "mrr": round(sum(mrr) / len(ranks), 3),
            "goal_mode_frac": round(goal_mode / len(queries), 3) if method == "adaptive" else None,
            "goal_mode_acc": round(goal_correct / goal_mode, 3) if goal_mode else None,
        }
        detail[name] = [{"target": TASKS[queries[i][3]][2], "rank": ranks[i]} for i in range(len(queries))]
        gm = results[name]["goal_mode_frac"]
        extra = f" goal_mode={gm} acc={results[name]['goal_mode_acc']}" if gm is not None else ""
        print(f"{name:20s} hit@6={results[name]['hit@6']} {results[name]['hit@6_ci']} mrr={results[name]['mrr']}{extra}", flush=True)

    out = Path(f"results/code/adaptive/{args.run_id}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"layer": args.layer, "k": args.k, "n_queries": len(queries), "metrics": results, "detail": detail},
        indent=2, ensure_ascii=False))
    lines = ["# Confidence-adaptive routing (goal when confident, flat fallback)", "",
             f"layer={args.layer} k={args.k} queries={len(queries)} (centroid, uncentered)", "",
             "| condition | hit@1 | hit@3 | hit@6 [95% CI] | MRR | goal_mode frac | goal_mode acc |", "|---|---|---|---|---|---|---|"]
    for name, mm in results.items():
        lines.append(f"| {name} | {mm['hit@1']} | {mm['hit@3']} | {mm['hit@6']} {mm['hit@6_ci']} | {mm['mrr']} | {mm['goal_mode_frac']} | {mm['goal_mode_acc']} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
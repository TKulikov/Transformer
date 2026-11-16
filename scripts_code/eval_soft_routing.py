"""Soft routing + goal+need blend vs the goal-identification bottleneck.

The realism test showed generated-goal routing collapses (routing_accuracy 0.333,
hit@6 0.267 vs oracle 0.8). This tests two fixes the author intended:
  * soft routing — weighted search over top-m goal nodes (not greedy argmax);
  * goal+need blend — routing signal combines the goal with which solved subgoals
    are needed now (pure need collapses, pure generated goal misroutes).

Conditions (all on the uncentered goal tree — centering is not load-bearing for
code): oracle_greedy, gen_greedy, gen_greedy_blend, gen_soft_goalonly,
gen_soft_blend (m=3), gen_soft_blend_m6 (all goals), oracle_soft_blend.
Reports hit@k, MRR, bootstrap CI, greedy routing accuracy / soft coverage@3.
Saves results/code/soft_routing/<run-id>/. GPU run:

    HF_HUB_OFFLINE=1 .venv/bin/python scripts_code/eval_soft_routing.py --run-id s1
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
from codemem.vector import normalize
from eval_retrieval import (GOALS, TASKS, bootstrap_ci, build_goal, extract_functions,
                            metric_vec, mrr_vec)

# name, method, goal_source, need_weight, m
CONDS = [
    ("oracle_greedy",      "greedy", "oracle", 0.0, 1),
    ("gen_greedy",         "greedy", "gen",    0.0, 1),
    ("gen_greedy_blend",   "greedy", "gen",    0.5, 1),
    ("gen_soft_goalonly",  "soft",   "gen",    0.0, 3),
    ("gen_soft_blend",     "soft",   "gen",    0.5, 3),
    ("gen_soft_blend_m6",  "soft",   "gen",    0.5, 6),
    ("oracle_soft_blend",  "soft",   "oracle", 0.5, 3),
]


def blend(goal_emb, need_emb, need_weight):
    return normalize(normalize(goal_emb) + need_weight * normalize(need_emb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="s1")
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
    for name, method, src, nw, m in CONDS:
        ranks, route_ok = [], 0
        for i, (g, need, tid, ti) in enumerate(queries):
            goal_src = goal_embs[g] if src == "oracle" else gen_goal_embs[ti]
            ne = need_embs[i]
            if method == "greedy":
                route_emb = blend(goal_src, ne, nw) if nw > 0 else goal_src
                if tree.route(route_emb) == goal_node_id[g]:
                    route_ok += 1
                out = retr.retrieve(route_emb, ne, i, k=args.k)
            else:
                cov = blend(goal_src, ne, nw)
                if goal_node_id[g] in [nid for nid, _ in tree.route_soft(cov, m=m)]:
                    route_ok += 1
                out = retr.retrieve_soft(goal_src, ne, i, k=args.k, m=m, need_weight=nw)
            ranks.append(out.index(tid) + 1 if tid in out else None)
        h6, mrr = metric_vec(ranks, args.k), mrr_vec(ranks)
        results[name] = {
            "hit@1": round(sum(metric_vec(ranks, 1)) / len(ranks), 3),
            "hit@3": round(sum(metric_vec(ranks, 3)) / len(ranks), 3),
            "hit@6": round(sum(h6) / len(ranks), 3), "hit@6_ci": bootstrap_ci(h6),
            "mrr": round(sum(mrr) / len(ranks), 3),
            "route_acc_or_cov": round(route_ok / len(queries), 3),
        }
        detail[name] = [{"target": TASKS[queries[i][3]][2], "rank": ranks[i]} for i in range(len(queries))]
        print(f"{name:20s} hit@6={results[name]['hit@6']} {results[name]['hit@6_ci']} "
              f"mrr={results[name]['mrr']} route={results[name]['route_acc_or_cov']}", flush=True)

    out = Path(f"results/code/soft_routing/{args.run_id}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"layer": args.layer, "k": args.k, "n_queries": len(queries), "metrics": results, "detail": detail},
        indent=2, ensure_ascii=False))
    lines = ["# Soft routing + goal+need blend (vs goal-identification bottleneck)", "",
             f"layer={args.layer} k={args.k} queries={len(queries)} (uncentered goal tree)", "",
             "| condition | hit@1 | hit@3 | hit@6 [95% CI] | MRR | route_acc/coverage |", "|---|---|---|---|---|---|"]
    for name, mm in results.items():
        lines.append(f"| {name} | {mm['hit@1']} | {mm['hit@3']} | {mm['hit@6']} {mm['hit@6_ci']} | {mm['mrr']} | {mm['route_acc_or_cov']} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
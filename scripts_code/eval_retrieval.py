"""Phase-4 retrieval ablation on the taskq codebase (decomposed + stats).

Disentangles centering, MMR diversification, goal routing, the prefetch cascade,
need-based routing, and (with --realism) routing by a GENERATED goal instead of
the oracle goal — measuring how far practical routing falls below the upper bound.

Conditions (kind, centered, lambda, prefetch, route_mode):
  flat_plain / flat_diverse / flat_centered / flat_centered_diverse
  goal_uncentered / goal_centered / goal_prefetch
  needroute        (route by the need — collapses)
  goal_generated   (--realism: route by GoalBuilder's generated goal embedding)

Queries: each handcrafted need + 2 model paraphrases (n=45). Reports hit@1/3/6,
MRR, 95% bootstrap CI, paired never-worse contrasts, and routing accuracy for
goal_generated. Saves results/code/retrieval/<run-id>/. GPU run:

    HF_HUB_OFFLINE=1 .venv/bin/python scripts_code/eval_retrieval.py --run-id r2
    HF_HUB_OFFLINE=1 .venv/bin/python scripts_code/eval_retrieval.py --run-id realism --realism
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from codemem.bank import CodeBank
from codemem.config import Config
from codemem.goal import GoalBuilder
from codemem.model import load_model
from codemem.prefetch import PrefetchBank
from codemem.retrieve import Retriever
from codemem.tree import GoalTree

CORPUS = Path("testbeds/taskq/taskq")

GOALS = {
    "auth": "authentication and token validation",
    "storage": "in-memory task store and results persistence",
    "scheduling": "priority scheduling and batching of tasks",
    "networking": "wire protocol, connections and message framing",
    "middleware": "request middleware and rate limiting",
    "utils": "shared cross-cutting helpers and utilities",
}
MODULE_GOAL = {
    "auth": "auth", "storage": "storage", "filestore": "storage",
    "scheduler": "scheduling", "worker": "scheduling",
    "server": "networking", "connection": "networking", "protocol": "networking",
    "router": "networking", "client": "networking",
    "middleware": "middleware", "ratelimit": "middleware", "handlers": "middleware",
    "utils": "utils", "errors": "utils", "config": "utils", "models": "utils",
    "cli": "utils", "__init__": "utils",
}

TASKS = [
    ("auth", "reject a request whose token is not in the allowed set", "auth.AuthMiddleware.before"),
    ("storage", "store a task result and mark the task finished", "storage.TaskStore.save_result"),
    ("storage", "get all tasks still waiting to run, oldest first", "storage.TaskStore.pending"),
    ("storage", "update a task status and bump its attempt counter", "storage.TaskStore.set_status"),
    ("scheduling", "move failed tasks back to pending for another attempt", "scheduler.Scheduler.requeue_failed"),
    ("scheduling", "return up to batch_size pending tasks to run", "scheduler.Scheduler.next_batch"),
    ("networking", "serialize a message to a length-prefixed JSON byte frame", "protocol.encode_message"),
    ("networking", "parse a length-prefixed JSON frame into a message", "protocol.decode_message"),
    ("middleware", "check whether a key is within its sliding-window rate budget", "ratelimit.RateLimiter.allow"),
    ("utils", "exponential backoff delay in ms for a zero-based attempt index", "utils.backoff_delay"),
    ("utils", "call a function with exponential backoff and retry on failure", "utils.retry"),
    ("utils", "parse a duration like 500ms or 2s into milliseconds", "utils.parse_duration"),
    ("utils", "stable deduplication preserving first-occurrence order", "utils.dedupe"),
    ("utils", "clamp a value into an inclusive range", "utils.clamp"),
    ("utils", "yield successive fixed-size chunks from a sequence", "utils.chunked"),
]

# name -> (kind, centered, lambda, prefetch, route_mode)
CONDITIONS = {
    "flat_plain":            ("flat", False, 1.0, False, "goal"),
    "flat_diverse":          ("flat", False, 0.5, False, "goal"),
    "flat_centered":         ("flat", True, 1.0, False, "goal"),
    "flat_centered_diverse": ("flat", True, 0.5, False, "goal"),
    "goal_uncentered":       ("goal", False, 0.5, False, "goal"),
    "goal_centered":         ("goal", True, 0.5, False, "goal"),
    "goal_prefetch":         ("goal", True, 0.5, True, "goal"),
    "needroute":             ("goal", True, 0.5, False, "need"),
}


def extract_functions():
    out = []
    for path in sorted(CORPUS.glob("*.py")):
        module = path.stem
        src = path.read_text()
        tree = ast.parse(src)
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                out.append((f"{module}.{cls.name}.{fn.name}", module, ast.get_source_segment(src, fn)))
        for fn in [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            out.append((f"{module}.{fn.name}", module, ast.get_source_segment(src, fn)))
    return out


def assign(tree, bank, node_id, cid):
    node = tree.nodes[node_id]
    node.chunk_ids.append(cid)
    node.centroid.update(bank.get(cid).emb)
    bank.get(cid).goal_id = node_id


def build_flat(bank, ids, centered):
    tree = GoalTree(bank, centered=centered, mmr_lambda=0.5)
    root = tree.add_goal("taskq", torch.zeros(bank.get(ids[0]).emb.shape))
    for cid in ids:
        assign(tree, bank, root, cid)
    return tree


def build_goal(bank, chunks, goal_embs, centered):
    tree = GoalTree(bank, centered=centered, mmr_lambda=0.5)
    root = tree.add_goal("task queue server", torch.zeros(bank.get(chunks[0][0]).emb.shape))
    nodes = {g: tree.add_goal(GOALS[g], emb, parent=root) for g, emb in goal_embs.items()}
    for cid, module, _ in chunks:
        assign(tree, bank, nodes[MODULE_GOAL.get(module, "utils")], cid)
    return tree


def bootstrap_ci(values, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)], 3), round(means[int(0.975 * n_boot)], 3)


def metric_vec(ranks, kk):
    return [1.0 if (r is not None and r <= kk) else 0.0 for r in ranks]


def mrr_vec(ranks):
    return [1.0 / r if r is not None else 0.0 for r in ranks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="r2")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--paraphrases", type=int, default=2)
    ap.add_argument("--realism", action="store_true", help="add goal_generated routing condition")
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
    chunks = []
    for (qname, module, src), emb in zip(funcs, embs):
        chunks.append((bank.add(src, emb), module, qname))
    qname_to_id = {q: cid for cid, _, q in chunks}
    ids = [cid for cid, _, _ in chunks]
    print(f"corpus: {len(funcs)} functions", flush=True)

    goal_embs = {g: model.embed([desc])[0] for g, desc in GOALS.items()}

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
    print(f"queries: {len(queries)} ({len(TASKS)} tasks x {args.paraphrases+1})", flush=True)

    conditions = dict(CONDITIONS)
    gen_goal_embs = {}
    if args.realism:
        conditions["goal_generated"] = ("goal", True, 0.5, False, "generated")
        gb = GoalBuilder(model, cfg)
        for ti, (g, need, target) in enumerate(TASKS):
            gen_goal_embs[ti] = gb.build(f"I need to {need}").emb
        print(f"generated goals for {len(gen_goal_embs)} tasks", flush=True)

    trees = {
        ("flat", False): build_flat(bank, ids, False),
        ("flat", True): build_flat(bank, ids, True),
        ("goal", False): build_goal(bank, chunks, goal_embs, False),
        ("goal", True): build_goal(bank, chunks, goal_embs, True),
    }
    goal_node_id = {}
    for nid, n in trees[("goal", True)].nodes.items():
        for g, desc in GOALS.items():
            if n.text == desc:
                goal_node_id[g] = nid

    results, detail, rank_table = {}, {}, {}
    for name, (kind, centered, lam, use_pf, route_mode) in conditions.items():
        tree = trees[(kind, centered)]
        pf = PrefetchBank(ttl_steps=1)
        retr = Retriever(tree, bank, pf, top_k=args.k, mmr_lambda=lam, centered=centered)
        ranks, route_correct = [], 0
        for i, (g, need, tid, ti) in enumerate(queries):
            if route_mode == "need":
                route_emb = need_embs[i]
            elif route_mode == "generated":
                route_emb = gen_goal_embs[ti]
            else:
                route_emb = goal_embs[g]
            if route_mode == "generated" and tree.route(route_emb) == goal_node_id[g]:
                route_correct += 1
            if use_pf:
                node = tree.route(route_emb)
                pf.put(i, tree.search(node, need_embs[i], 8, lambda_=1.0))
            out = retr.retrieve(route_emb, need_embs[i], i, k=args.k)
            ranks.append(out.index(tid) + 1 if tid in out else None)
        rank_table[name] = ranks
        h6, mrr = metric_vec(ranks, args.k), mrr_vec(ranks)
        results[name] = {
            "hit@1": round(sum(metric_vec(ranks, 1)) / len(ranks), 3),
            "hit@3": round(sum(metric_vec(ranks, 3)) / len(ranks), 3),
            "hit@6": round(sum(h6) / len(ranks), 3), "hit@6_ci": bootstrap_ci(h6),
            "mrr": round(sum(mrr) / len(ranks), 3), "mrr_ci": bootstrap_ci(mrr),
        }
        if route_mode == "generated":
            results[name]["routing_accuracy"] = round(route_correct / len(queries), 3)
        detail[name] = [{"need": queries[i][1], "target": TASKS[queries[i][3]][2], "rank": ranks[i]} for i in range(len(queries))]
        extra = f" route_acc={results[name].get('routing_accuracy')}" if route_mode == "generated" else ""
        print(f"{name:22s} hit@6={results[name]['hit@6']} {results[name]['hit@6_ci']} mrr={results[name]['mrr']}{extra}", flush=True)

    def score(r):
        return r if r is not None else 10**6
    contrasts = {
        "diversification_flat": ("flat_plain", "flat_diverse"),
        "centering_flat": ("flat_plain", "flat_centered"),
        "routing": ("flat_centered_diverse", "goal_centered"),
        "centering_goal": ("goal_uncentered", "goal_centered"),
        "prefetch": ("goal_centered", "goal_prefetch"),
        "oracle_vs_needroute": ("goal_centered", "needroute"),
    }
    if args.realism:
        contrasts["oracle_vs_generated"] = ("goal_centered", "goal_generated")
    paired = {}
    for cname, (a, b) in contrasts.items():
        ra, rb = rank_table[a], rank_table[b]
        better = sum(1 for x, y in zip(ra, rb) if score(y) < score(x))
        worse = sum(1 for x, y in zip(ra, rb) if score(y) > score(x))
        paired[cname] = {"baseline": a, "treatment": b, "better": better, "worse": worse, "equal": len(ra) - better - worse}
        print(f"paired {cname:24s} {b} vs {a}: +{better}/-{worse}/={len(ra)-better-worse}", flush=True)

    out = Path(f"results/code/retrieval/{args.run_id}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"layer": args.layer, "k": args.k, "n_funcs": len(funcs), "n_queries": len(queries),
         "metrics": results, "paired": paired, "detail": detail}, indent=2, ensure_ascii=False))
    lines = ["# Retrieval ablation (taskq, decomposed + bootstrap)", "",
             f"layer={args.layer} k={args.k} corpus={len(funcs)} queries={len(queries)}", "",
             "| condition | hit@1 | hit@3 | hit@6 [95% CI] | MRR [95% CI] | route_acc |", "|---|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['hit@1']} | {m['hit@3']} | {m['hit@6']} {m['hit@6_ci']} | {m['mrr']} {m['mrr_ci']} | {m.get('routing_accuracy','-')} |")
    lines += ["", "## Paired contrasts (treatment vs baseline, per query)", "",
              "| contrast | better | worse | equal |", "|---|---|---|---|"]
    for cname, p in paired.items():
        lines.append(f"| {cname} ({p['treatment']} vs {p['baseline']}) | {p['better']} | {p['worse']} | {p['equal']} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
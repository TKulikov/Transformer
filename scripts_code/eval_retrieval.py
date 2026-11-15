"""Phase-4 retrieval ablation on the taskq codebase (decomposed + stats).

Disentangles the three mechanisms that the dialogue work found load-bearing —
centering, MMR diversification, and goal routing — plus the prefetch cascade and
realistic (need-based, non-oracle) routing. Conditions:

  flat_plain            flat, no centering, pure relevance (lam=1)
  flat_diverse          flat, no centering, MMR (lam=0.5)
  flat_centered         flat, global centering, pure relevance
  flat_centered_diverse flat, global centering, MMR
  goal_uncentered       goal tree, no centering, MMR
  goal_centered         goal tree, local centering, MMR        [workhorse]
  goal_prefetch         goal_centered + prefetch unified cascade
  needroute             goal tree routed by the NEED (no oracle goal), MMR

Queries: each handcrafted need + 2 model paraphrases (n=45) to give statistical
power (retrieval is deterministic, so variance comes from query variation).
Reports hit@1/3/6, MRR, 95% bootstrap CI over queries, and paired never-worse
counts for key contrasts. Saves results/code/retrieval/<run-id>/. GPU run:

    HF_HUB_OFFLINE=1 .venv/bin/python scripts_code/eval_retrieval.py --run-id r2
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

# name -> (kind, centered, lambda, prefetch, route_by_need)
CONDITIONS = {
    "flat_plain":            ("flat", False, 1.0, False, False),
    "flat_diverse":          ("flat", False, 0.5, False, False),
    "flat_centered":         ("flat", True, 1.0, False, False),
    "flat_centered_diverse": ("flat", True, 0.5, False, False),
    "goal_uncentered":       ("goal", False, 0.5, False, False),
    "goal_centered":         ("goal", True, 0.5, False, False),
    "goal_prefetch":         ("goal", True, 0.5, True, False),
    "needroute":             ("goal", True, 0.5, False, True),
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
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
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

    # build query set: original need + paraphrases
    queries = []   # (goal, need_text, target_id, task_idx)
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

    trees = {
        ("flat", False): build_flat(bank, ids, False),
        ("flat", True): build_flat(bank, ids, True),
        ("goal", False): build_goal(bank, chunks, goal_embs, False),
        ("goal", True): build_goal(bank, chunks, goal_embs, True),
    }

    results, detail, rank_table = {}, {}, {}
    for name, (kind, centered, lam, use_pf, by_need) in CONDITIONS.items():
        tree = trees[(kind, centered)]
        pf = PrefetchBank(ttl_steps=1)
        retr = Retriever(tree, bank, pf, top_k=args.k, mmr_lambda=lam, centered=centered)
        ranks = []
        for i, (g, need, tid, ti) in enumerate(queries):
            route_emb = need_embs[i] if by_need else goal_embs[g]
            if use_pf:
                node = tree.route(route_emb)
                pf.put(i, tree.search(node, need_embs[i], 8, lambda_=1.0))
            out = retr.retrieve(route_emb, need_embs[i], i, k=args.k)
            ranks.append(out.index(tid) + 1 if tid in out else None)
        rank_table[name] = ranks
        h6 = metric_vec(ranks, args.k)
        mrr = mrr_vec(ranks)
        results[name] = {
            "hit@1": round(sum(metric_vec(ranks, 1)) / len(ranks), 3),
            "hit@3": round(sum(metric_vec(ranks, 3)) / len(ranks), 3),
            "hit@6": round(sum(h6) / len(ranks), 3),
            "hit@6_ci": bootstrap_ci(h6),
            "mrr": round(sum(mrr) / len(ranks), 3),
            "mrr_ci": bootstrap_ci(mrr),
        }
        detail[name] = [{"need": queries[i][1], "target": TASKS[queries[i][3]][2], "rank": ranks[i]} for i in range(len(queries))]
        print(f"{name:22s} hit@6={results[name]['hit@6']} {results[name]['hit@6_ci']} mrr={results[name]['mrr']}", flush=True)

    # paired never-worse contrasts (per query rank; lower=better, None=miss=worst)
    def score(r):
        return r if r is not None else 10**6
    contrasts = {
        "diversification_flat": ("flat_plain", "flat_diverse"),
        "centering_flat": ("flat_plain", "flat_centered"),
        "centering_flat_diverse": ("flat_diverse", "flat_centered_diverse"),
        "routing": ("flat_centered_diverse", "goal_centered"),
        "centering_goal": ("goal_uncentered", "goal_centered"),
        "prefetch": ("goal_centered", "goal_prefetch"),
        "oracle_vs_needroute": ("goal_centered", "needroute"),
    }
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
             "| condition | hit@1 | hit@3 | hit@6 [95% CI] | MRR [95% CI] |", "|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['hit@1']} | {m['hit@3']} | {m['hit@6']} {m['hit@6_ci']} | {m['mrr']} {m['mrr_ci']} |")
    lines += ["", "## Paired contrasts (treatment vs baseline, per query)", "",
              "| contrast | better | worse | equal |", "|---|---|---|---|"]
    for cname, p in paired.items():
        lines.append(f"| {cname} ({p['treatment']} vs {p['baseline']}) | {p['better']} | {p['worse']} | {p['equal']} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
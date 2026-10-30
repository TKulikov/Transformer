"""Benchmark CLI: run the strategy grid on a real model and save results.

Examples:
    python -m tmem.cli --model gpt2 --device cpu
    python -m tmem.cli --model qwen2-1.5b --device cuda --budget 128 --out results/qwen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .dialogue import default_strategies, run_session
from .models import load_model
from .plotting import plot_cache_growth, plot_context_growth, plot_efficiency_quality
from .qa import GENERAL_QA, SESSIONS, score_qa_answer


def answer_questions(bundle, *, kv, cache="use", max_new_tokens=32, seed=0) -> dict:
    """Answer the standalone general-knowledge questions and score them."""
    results = {}
    for i, item in enumerate(GENERAL_QA):
        text = f"Question: {item.question} Answer:"
        ids = bundle.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(bundle.device)
        common = dict(kv=kv, cache=cache, max_new_tokens=max_new_tokens, do_sample=False, seed=seed + i)
        if bundle.kind == "causal":
            from .generate import decode_causal

            out = decode_causal(bundle.model, ids, eos_ids=bundle.eos_ids(), **common)
        else:
            from .generate import decode_encdec

            out = decode_encdec(bundle.model, ids, decoder_start_id=bundle.decoder_start_id(), eos_ids=(1,), **common)
        answer = bundle.tokenizer.decode(out.token_ids, skip_special_tokens=True).strip()
        results[item.question] = {"answer": answer[:200], "expected": item.expected, "score": score_qa_answer(answer, item)}
    return results


def run_stress(bundle, args, sessions=None, strategies=None, tag="stress") -> dict:
    """Big stress battery: sessions × condition grid × analysis, saved as JSON."""
    from .stress import analyze_session, stress_sessions, stress_strategies

    sessions = sessions or stress_sessions()
    strategies = strategies or stress_strategies(budget=args.budget)
    report = {"model": bundle.name, "device": str(bundle.device), "tag": tag, "sessions": {}}
    for session in sessions:
        runs = {
            s.name: run_session(bundle, session, s, max_new_tokens=args.max_new_tokens, seed=args.seed)
            for s in strategies
        }
        analysis = analyze_session(runs)
        report["sessions"][session.name] = {
            "turns": len(session.user_turns()),
            "n_probes": len(session.probes),
            "strategies": analysis,
        }
        best = max(analysis, key=lambda k: analysis[k]["recall_mean"])
        print(f"[{tag}:{bundle.name}] session={session.name} best_recall={best}({analysis[best]['recall_mean']})")
        flat = {n: r for n, r in runs.items() if n in ("classic_kv_full", "v2_rag_gist", "v2_full", "window2_only", "v1_memory")}
        from .plotting import plot_context_growth, plot_efficiency_quality

        out_dir = Path(args.out)
        plot_context_growth(flat, str(out_dir / f"{bundle.name}_{session.name}_context.png"))
        plot_efficiency_quality(flat, str(out_dir / f"{bundle.name}_{session.name}_quality.png"))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="gpt2 | qwen2-1.5b | t5-base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--budget", type=int, default=128, help="sparse KV-cache budget (slots)")
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--recent", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategies", nargs="*", default=None, help="subset of strategy names")
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--sessions", nargs="*", default=None, help="subset of session names")
    parser.add_argument("--stress", action="store_true", help="run the big memory stress battery instead of the standard benchmark")
    parser.add_argument("--battery100", action="store_true", help="run the >100-turn battery (facts/updates/probes across zones)")
    parser.add_argument("--models", nargs="*", default=None, help="multiple models for --stress (overrides --model)")
    args = parser.parse_args(argv)

    dtype = {"auto": "auto", "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stress or args.battery100:
        from .stress import build_battery_100, battery_strategies

        tag = "battery100" if args.battery100 else "stress"
        for mname in (args.models or [args.model]):
            b = load_model(mname, device=args.device, dtype=dtype, eager=not args.battery100)
            if args.battery100:
                strats = battery_strategies(budget=args.budget)
                if args.strategies:
                    strats = [s for s in strats if s.name in args.strategies]
                rep = run_stress(b, args, sessions=[build_battery_100()], strategies=strats, tag=tag)
            else:
                rep = run_stress(b, args)
            (out_dir / f"{mname}_{tag}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
            print(f"[{tag}] {mname}: results json written")
            del b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return 0

    bundle = load_model(args.model, device=args.device, dtype=dtype, eager=True)

    strategies = default_strategies(budget=args.budget, sink=args.sink, recent=args.recent)
    if args.strategies:
        strategies = [s for s in strategies if s.name in args.strategies]

    report: dict = {"model": args.model, "device": str(bundle.device), "strategies": [s.describe() for s in strategies], "sessions": {}, "general_qa": {}}
    plot_runs: dict[str, object] = {}
    for strategy in strategies:
        first_session_run = None
        for session in SESSIONS:
            if args.sessions and session.name not in args.sessions:
                continue
            run = run_session(bundle, session, strategy, max_new_tokens=args.max_new_tokens, seed=args.seed)
            if first_session_run is None:
                first_session_run = run
                plot_runs[strategy.name] = run
            entry = report["sessions"].setdefault(session.name, {})
            entry[strategy.name] = {
                "summary": run.summary(),
                "answers": [
                    {"turn": i + 1, "user": t.user_text, "answer": t.answer_text[:160], "probe_score": t.probe_score}
                    for i, t in enumerate(run.turns)
                ],
            }
        report["general_qa"][strategy.name] = answer_questions(bundle, kv=strategy.kv, cache=strategy.cache, seed=args.seed)
        print(f"[done] strategy={strategy.name}")

    if plot_runs:
        plot_context_growth(plot_runs, str(out_dir / f"{args.model}_context.png"))
        plot_cache_growth(plot_runs, str(out_dir / f"{args.model}_cache.png"))
        plot_efficiency_quality(plot_runs, str(out_dir / f"{args.model}_quality.png"))

    path = out_dir / f"{args.model}_benchmark.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"results written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

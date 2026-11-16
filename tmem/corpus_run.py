"""Batch runner for the multi-dialogue corpus benchmark (plan step 2).

Execution model (approved in docs/PLAN-fullscale-tests.md):

* **batch = one dialogue × all strategies** — atomic and resumable;
* run state lives on disk: ``run_config.json`` (frozen queue),
  ``manifest.jsonl`` (one line per finished/failed batch),
  ``batches/<batch_id>/<strategy>.json``;
* after every batch ``INTERIM.md`` is regenerated from the same statistics
  kernel used for the final aggregate (:mod:`tmem.corpus_stats`);
* the GPU is held only for ``--batches N`` invocations; stopping between
  batches loses nothing, stopping inside a batch loses that batch only.

Latency protocol: repetition 0 is a warmup whose answers provide quality
scores (greedy decoding is deterministic); repetitions 1..R are interleaved
across strategies (A,B,A,B,…) and provide end-to-end wall time **including
embedding and retrieval** (whole ``run_session`` call), with strategy order
rotated between repetitions and batches to cancel GPU drift.

Usage:
    python -m tmem.corpus_run --preset pilot --model qwen35-4b --device cuda \
        --dtype bfloat16 --batches 4 --run-id pilot-v1
    python -m tmem.corpus_run --preset main --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from .corpus import PRESETS, CorpusSpec, build_corpus_session, specs_from_json, validate_session
from .corpus_stats import render_interim_md
from .dialogue import Strategy, run_session
from .kvcache import KVConfig
from .stress import analyze_session


# ---------------------------------------------------------------- strategies


def main_strategies(budget: int = 128) -> list[Strategy]:
    """The seven comparable systems of the paper (same names as battery100)."""
    return [
        Strategy("classic_kv_full", context="full"),
        Strategy(f"sparse_sliding_{budget}", context="full", kv=KVConfig.sparse(budget, "sliding", 4, 64)),
        Strategy(f"sparse_topk_{budget}", context="full", kv=KVConfig.sparse(budget, "topk", 4, 64)),
        Strategy(f"snapkv_{budget}", context="full", kv=KVConfig.sparse(budget, "snapkv", observation=64, kernel_size=7)),
        Strategy(f"pyramidkv_{budget}", context="full", kv=KVConfig.sparse(budget, "pyramidkv", observation=8, pyramid_beta=20)),
        Strategy("rag_text_only", context="window", window=5, rag=True, rag_lambda=0.0, rag_centered=False),
        Strategy("v4_rag_kv", context="window", window=5, memory_kv=True, rag=True, rag_lambda=0.0,
                 rag_centered=True, memory_window_crit=0.5, memory_crit=0.95, memory_merge_k=0.25, kv_slots=32),
    ]


def ablation_strategies(budget: int = 128) -> list[Strategy]:
    """Component ladder: RAG → +upsert → +centering → +latent K/V (and latent-only)."""
    base = dict(context="window", window=5, rag=True, rag_lambda=0.0)
    mem = dict(memory_window_crit=0.5, memory_crit=0.95, memory_merge_k=0.25)
    return [
        Strategy("classic_kv_full", context="full"),
        Strategy("abl_rag_plain", **base, rag_crit=1.1, rag_centered=False),                      # no upsert, no centering
        Strategy("abl_rag_upsert", **base, rag_crit=0.95, rag_centered=False),                   # + upsert
        Strategy("abl_rag_centered_upsert", **base, rag_crit=0.95, rag_centered=True),           # + centering
        Strategy("abl_kv_only", context="window", window=5, memory_kv=True, kv_slots=32, **mem),  # latent channel only
        Strategy("abl_rag_kv", **base, rag_crit=0.95, rag_centered=True, memory_kv=True, kv_slots=32, **mem),
    ]


def sweep_strategies(budget: int = 128) -> list[Strategy]:
    """Hyper-parameter sweeps around the final system and the KV controls."""
    mem = dict(memory_window_crit=0.5, memory_crit=0.95, memory_merge_k=0.25)
    base = dict(context="window", window=5, rag=True, rag_lambda=0.0, rag_centered=True, rag_crit=0.95)
    out = [Strategy("classic_kv_full", context="full")]
    out += [Strategy(f"v4_slots{k}", **base, memory_kv=True, kv_slots=k, **mem) for k in (8, 16, 32, 64)]
    out += [Strategy(f"v4_ragk{k}", **{**base, "rag_k": k}, memory_kv=True, kv_slots=32, **mem) for k in (2, 4, 8)]
    for b in (64, 256):
        out += [
            Strategy(f"sparse_sliding_{b}", context="full", kv=KVConfig.sparse(b, "sliding", 4, max(8, b // 2))),
            Strategy(f"sparse_topk_{b}", context="full", kv=KVConfig.sparse(b, "topk", 4, max(8, b // 2))),
        ]
    return out


def latent_strategies(budget: int = 128) -> list[Strategy]:
    """New-version campaign: the latent (mid-layer) bank both feeds the K/V
    injection and keys the RAG retrieval. Old v4 (last-layer RAG keys) and the
    KV/RAG analogues run alongside on the fixed kvinject for direct comparison;
    rag_latent_mid and v4_kv_mid isolate the two halves of the new system."""
    mem = dict(memory_window_crit=0.5, memory_crit=0.95, memory_merge_k=0.25)
    return [
        Strategy("classic_kv_full", context="full"),
        Strategy(f"sparse_sliding_{budget}", context="full", kv=KVConfig.sparse(budget, "sliding", 4, 64)),
        Strategy(f"sparse_topk_{budget}", context="full", kv=KVConfig.sparse(budget, "topk", 4, 64)),
        Strategy(f"snapkv_{budget}", context="full", kv=KVConfig.sparse(budget, "snapkv", observation=64, kernel_size=7)),
        Strategy(f"pyramidkv_{budget}", context="full", kv=KVConfig.sparse(budget, "pyramidkv", observation=8, pyramid_beta=20)),
        Strategy("rag_text_only", context="window", window=5, rag=True, rag_lambda=0.0, rag_centered=False),
        Strategy("rag_latent_mid", context="window", window=5, rag=True, rag_lambda=0.0,
                 rag_centered=False, rag_layer="mid"),
        Strategy("v4_rag_kv", context="window", window=5, memory_kv=True, rag=True, rag_lambda=0.0,
                 rag_centered=True, kv_slots=32, **mem),
        Strategy("v4_rag_kv_mid", context="window", window=5, memory_kv=True, rag=True, rag_lambda=0.0,
                 rag_centered=True, rag_layer="mid", kv_slots=32, **mem),
        Strategy("v4_kv_mid", context="window", window=5, memory_kv=True, kv_slots=32, **mem),
    ]


def shared_strategies(budget: int = 128) -> list[Strategy]:
    """The shared-bank hypothesis: RAG keyed by the latent bank's own slot
    vectors — the same CorrectionLayer activations that are poured into the
    bank and injected into attention (rag_key_source="bank"). classic_kv_full
    is the per-run gold reference for cross-run drift checks."""
    mem = dict(memory_window_crit=0.5, memory_crit=0.95, memory_merge_k=0.25)
    return [
        Strategy("classic_kv_full", context="full"),
        Strategy("rag_bank_key", context="window", window=5, rag=True, rag_lambda=0.0,
                 rag_centered=False, rag_key_source="bank", **mem),
        Strategy("v4_rag_kv_bank", context="window", window=5, memory_kv=True, rag=True,
                 rag_lambda=0.0, rag_centered=True, rag_key_source="bank", kv_slots=32, **mem),
    ]


def shared_dev_strategies(budget: int = 128) -> list[Strategy]:
    """Anchored bank-dev campaign. Two comparisons on the same dialogues:
    RAG-vs-RAG (utterance keys vs clustered bank-slot keys, ± injection) and
    living-memory-vs-PyramidKV without RAG — the latter with a bank-size
    sweep (the CorrectionLayer size, injection width fixed at kv_slots=32)."""
    mem = dict(memory_window_crit=0.5, memory_crit=0.95, memory_merge_k=0.25)
    return [
        Strategy("classic_kv_full", context="full"),
        Strategy("rag_bank_dev", context="window", window=5, rag=True, rag_lambda=0.0,
                 rag_centered=False, rag_key_source="bank_dev", **mem),
        Strategy("v4_rag_kv_bank_dev", context="window", window=5, memory_kv=True, rag=True,
                 rag_lambda=0.0, rag_centered=True, rag_key_source="bank_dev", kv_slots=32, **mem),
        # living memory without RAG vs PyramidKV — bank-size sweep
        Strategy("v4_kv_mid", context="window", window=5, memory_kv=True, kv_slots=32, **mem),
        Strategy("v4_kv_mid_s128", context="window", window=5, memory_kv=True, kv_slots=32,
                 memory_size=128, **mem),
        Strategy("v4_kv_mid_s1024", context="window", window=5, memory_kv=True, kv_slots=32,
                 memory_size=1024, **mem),
        Strategy(f"pyramidkv_{budget}", context="full",
                 kv=KVConfig.sparse(budget, "pyramidkv", observation=8, pyramid_beta=20)),
    ]


STRATEGY_SETS = {
    "pilot": main_strategies,
    "main": main_strategies,
    "ablation": ablation_strategies,
    "sweep": sweep_strategies,
    "latent": latent_strategies,
    "shared": shared_strategies,
    "shared_dev": shared_dev_strategies,
}


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- model runner


class ModelBatchRunner:
    """Runs one batch (dialogue × strategies × repeats) on a loaded model bundle."""

    def __init__(self, bundle, strategies: list[Strategy], *, max_new_tokens: int = 32,
                 seed: int = 0, latency_repeats: int = 5):
        self.bundle = bundle
        self.strategies = strategies
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.latency_repeats = latency_repeats

    def __call__(self, spec: CorpusSpec) -> dict:
        session = build_corpus_session(spec)
        problems = validate_session(session)
        if problems:
            return {"errors": [f"invalid session: {p}" for p in problems]}

        walls: dict[str, list[float]] = {s.name: [] for s in self.strategies}
        quality_runs: dict[str, object] = {}
        offset = sum(spec.spec_id.encode()) % max(len(self.strategies), 1)
        total_reps = 1 + self.latency_repeats  # rep 0 = warmup + quality
        n = len(self.strategies)
        for rep in range(total_reps):
            rot = (rep + offset) % n
            order = self.strategies[rot:] + self.strategies[:rot]
            for strategy in order:
                t0 = time.perf_counter()
                run = run_session(self.bundle, session, strategy,
                                  max_new_tokens=self.max_new_tokens, seed=self.seed)
                wall = time.perf_counter() - t0
                if rep == 0:
                    quality_runs[strategy.name] = run
                else:
                    walls[strategy.name].append(wall)

        n_turns = max(len(session.user_turns()), 1)
        analysis = analyze_session(quality_runs)
        results = {}
        for name in analysis:
            per_rep = [w / n_turns for w in walls.get(name, [])]
            results[name] = {
                "analysis": analysis[name],
                "e2e": {
                    "per_rep_mean_s": [round(w, 4) for w in per_rep],
                    "mean_s": round(mean(per_rep), 4) if per_rep else None,
                    "median_s": round(median(per_rep), 4) if per_rep else None,
                    "includes_retrieval": True,
                    "warmup_reps": 1,
                },
            }
        return {"results": results, "errors": [], "n_probes": len(session.probes),
                "n_turns": len(session.user_turns())}


# ---------------------------------------------------------------- run engine


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_manifest(run_dir: Path) -> dict[str, dict]:
    """Latest manifest record per batch_id."""
    manifest = {}
    path = run_dir / "manifest.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            manifest[rec["batch_id"]] = rec
    return manifest


def open_run(run_dir: Path, *, specs: list[CorpusSpec], strategy_names: list[str],
             meta: dict) -> dict:
    """Create or resume a run directory; the first call freezes the queue."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        frozen = {s["spec_id"] for s in cfg["specs"]}
        for s in specs:  # a resumed run may only shrink the queue, not redefine it
            if s.spec_id not in frozen:
                raise SystemExit(
                    f"run_config в {run_dir} не содержит spec {s.spec_id}; "
                    f"используйте другой --run-id или удалите run_config.json"
                )
        return cfg
    cfg = {
        "created_at": _now(),
        "specs": [s.to_dict() for s in specs],
        "strategies": strategy_names,
        **meta,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    return cfg


def run_corpus(run_dir, *, specs: list[CorpusSpec], strategies: list[Strategy],
               runner, meta: dict, batches: int | None = None, only=None,
               dry_run: bool = False) -> dict:
    """Execute pending batches; regenerate INTERIM.md after each one.

    ``runner(spec) -> {"results": {strategy: {...}}, "errors": [...]}`` — the
    GPU-bound callable; tests inject a stub instead of a model.
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / "run_config.json"
    if dry_run:
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
        else:
            cfg = {"run_id": meta.get("run_id"), "preset": meta.get("preset"), "model": meta.get("model"),
                   "git_sha": meta.get("git_sha"), "protocol": meta.get("protocol"),
                   "specs": [s.to_dict() for s in specs], "strategies": [s.name for s in strategies]}
    else:
        cfg = open_run(run_dir, specs=specs, strategy_names=[s.name for s in strategies], meta=meta)
    queued = [CorpusSpec.from_dict(d) for d in cfg["specs"]]
    manifest = load_manifest(run_dir)
    if only:
        pending = [s for s in queued if s.spec_id in set(only)]
    else:
        pending = [s for s in queued if manifest.get(s.spec_id, {}).get("status") != "done"]
    selected = pending[:batches] if batches else pending

    done_count = sum(1 for m in manifest.values() if m["status"] == "done")
    if dry_run:
        print(f"run_id={cfg.get('run_id')} run_dir={run_dir}")
        print(f"выполнено: {done_count}/{len(queued)}; в очереди: {len(pending)}; к запуску сейчас: {len(selected)}")
        walls = [m.get("wall_s") or 0 for m in manifest.values() if m["status"] == "done"]
        if walls:
            print(f"средний батч: {mean(walls) / 60:.1f} мин; выбранные ≈ {mean(walls) * len(selected) / 3600:.2f} ч")
        for i, s in enumerate(selected, 1):
            print(f"  {i:3d}. {s.spec_id}")
        return {"done": 0, "skipped": done_count, "pending": len(pending)}

    (run_dir / "batches").mkdir(exist_ok=True)
    completed = 0
    for i, spec in enumerate(selected, 1):
        reruns = manifest.get(spec.spec_id, {}).get("reruns", 0)
        if spec.spec_id in manifest:
            reruns += 1
        started = time.perf_counter()
        print(f"[{cfg.get('run_id')}] batch {i}/{len(selected)}: {spec.spec_id}", flush=True)
        try:
            out = runner(spec)
        except KeyboardInterrupt:
            print("прервано пользователем — батч не завершён и будет перезапущен при resume", flush=True)
            raise
        except Exception as exc:  # a crashed batch must not kill the run
            out = {"errors": [f"{type(exc).__name__}: {exc}"]}
        wall = time.perf_counter() - started
        record = {
            "batch_id": spec.spec_id,
            "status": "failed" if out.get("errors") or not out.get("results") else "done",
            "started_at": _now(),
            "wall_s": round(wall, 2),
            "reruns": reruns,
            "spec": spec.to_dict(),
            "n_probes": out.get("n_probes"),
            "n_turns": out.get("n_turns"),
            "git_sha": meta.get("git_sha"),
            "model": meta.get("model"),
            "errors": out.get("errors", []),
        }
        if record["status"] == "done":
            batch_dir = run_dir / "batches" / spec.spec_id
            batch_dir.mkdir(parents=True, exist_ok=True)
            for name, payload in out["results"].items():
                (batch_dir / f"{name}.json").write_text(
                    json.dumps({"strategy": name, "spec_id": spec.spec_id, **payload},
                               indent=2, ensure_ascii=False)
                )
        with (run_dir / "manifest.jsonl").open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest[spec.spec_id] = record
        completed += record["status"] == "done"
        (run_dir / "INTERIM.md").write_text(render_interim_md(run_dir))
        print(f"[{cfg.get('run_id')}] batch {spec.spec_id}: {record['status']} за {wall / 60:.1f} мин", flush=True)
    return {"done": completed, "skipped": done_count, "pending": len(pending) - len(selected)}


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default=None, choices=sorted(PRESETS), help="corpus preset")
    parser.add_argument("--specs", default=None, help="custom spec list (JSON), overrides --preset")
    parser.add_argument("--run-id", default=None, help="run directory name (default: <preset>-v1)")
    parser.add_argument("--out", default="results/corpus", help="parent directory of run directories")
    parser.add_argument("--model", default=None, help="model name for tmem.models.load_model")
    parser.add_argument("--device", default="auto",
                        help="cpu | cuda | auto (ROCm builds also use 'cuda': HIP exposes AMD GPUs "
                             "under torch.cuda — 'auto' picks cuda when available, else cpu)")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latency-repeats", type=int, default=5, help="measured reps after the warmup rep")
    parser.add_argument("--strategies", nargs="*", default=None, help="subset of preset strategy names")
    parser.add_argument("--strategy-set", default=None, choices=sorted(STRATEGY_SETS),
                        help="strategy set override (default: by --preset)")
    parser.add_argument("--batches", type=int, default=None, help="run at most N pending batches, then stop")
    parser.add_argument("--only", nargs="*", default=None, help="(re)run exactly these batch ids")
    parser.add_argument("--dry-run", action="store_true", help="print queue/progress/ETA without GPU")
    args = parser.parse_args(argv)

    if not args.preset and not args.specs:
        parser.error("нужен --preset или --specs")
    preset = args.preset or "custom"
    specs = specs_from_json(args.specs) if args.specs else PRESETS[args.preset]()
    strategies = STRATEGY_SETS.get(args.strategy_set or args.preset, main_strategies)(budget=args.budget)
    if args.strategies:
        strategies = [s for s in strategies if s.name in args.strategies]
    run_id = args.run_id or f"{preset}-v1"
    run_dir = Path(args.out) / run_id

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else None

    meta = {
        "run_id": run_id,
        "preset": preset,
        "model": args.model,
        "device": device,
        "gpu": gpu_name,
        "torch": torch.__version__,           # e.g. "2.9.1+rocm6.3" — environment-drift audit
        "dtype": args.dtype,
        "git_sha": git_sha(),
        "protocol": {
            "decoding": "greedy",
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "kv_budget": args.budget,
            "latency_repeats": args.latency_repeats,
            "warmup_reps": 1,
            "latency": "end-to-end wall time of run_session incl. embedding/retrieval",
        },
    }

    if args.dry_run:
        print(f"device: {device}" + (f" ({gpu_name})" if gpu_name else "") + f", torch {torch.__version__}")
        runner = None
    else:
        if not args.model:
            parser.error("--model обязателен без --dry-run")
        from .models import load_model

        dtype = {"auto": "auto", "float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}[args.dtype]
        bundle = load_model(args.model, device=device, dtype=dtype)
        runner = ModelBatchRunner(bundle, strategies, max_new_tokens=args.max_new_tokens,
                                  seed=args.seed, latency_repeats=args.latency_repeats)

    summary = run_corpus(run_dir, specs=specs, strategies=strategies, runner=runner,
                         meta=meta, batches=args.batches, only=args.only, dry_run=args.dry_run)
    print(f"готово: done={summary['done']} skipped={summary['skipped']} pending={summary['pending']}")
    print(f"промежуточный отчёт: {run_dir / 'INTERIM.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Statistics kernel shared by the interim report and the final aggregation.

Plan: docs/PLAN-fullscale-tests.md, steps 2/5. Everything is computed from
per-dialogue values (bootstrap over dialogues, NOT over questions — questions
within a dialogue are dependent). The same functions render INTERIM.md after
every batch and the final AGGREGATE.md, so partial numbers never differ
methodologically from final ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

import numpy as np

KINDS = ("direct", "paraphrase", "multihop", "temporal", "unanswerable")
LAGS = ("0-5", "6-12", "13-20", "21+")


# ---------------------------------------------------------------- primitives


def bootstrap_ci(values, iters: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean over per-dialogue values."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return (None, None)
    if v.size == 1:
        return (round(float(v[0]), 4), round(float(v[0]), 4))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(iters, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (round(float(lo), 4), round(float(hi), 4))


def paired_bootstrap_p(a, b, iters: int = 4000, seed: int = 0):
    """Two-sided paired-bootstrap p-value for H0: mean(a) == mean(b).

    ``a`` and ``b`` are per-dialogue metric values in the same order.
    p = 2 * min(P(diff* <= 0), P(diff* >= 0)) over bootstrap resamples.
    """
    arr_a = np.asarray([x for x in a if x is not None], dtype=float)
    arr_b = np.asarray([x for x in b if x is not None], dtype=float)
    n = min(arr_a.size, arr_b.size)
    if n < 2:
        return None
    d = arr_a[:n] - arr_b[:n]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    means = d[idx].mean(axis=1)
    p_le = float((means <= 0).mean())
    p_ge = float((means >= 0).mean())
    return round(min(1.0, 2 * min(p_le, p_ge)), 4)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        out[k] = round(adj, 4)
        prev = adj
    return out


# ---------------------------------------------------------------- run loading


def load_run(run_dir) -> dict:
    """Load run_config + all completed batches of a corpus run directory."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    records_by_id: dict[str, dict] = {}
    manifest_path = run_dir / "manifest.jsonl"
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            rec = json.loads(line)
            records_by_id[rec["batch_id"]] = rec  # last record wins (reruns)
    records = []
    for rec in records_by_id.values():
        if rec.get("status") != "done":
            continue
        strategies = {}
        batch_dir = run_dir / "batches" / rec["batch_id"]
        if batch_dir.exists():
            for f in sorted(batch_dir.glob("*.json")):
                data = json.loads(f.read_text())
                strategies[data["strategy"]] = data
        if strategies:
            rec["strategies"] = strategies
            records.append(rec)
    return {"dir": run_dir, "config": cfg, "records": records}


def _analysis(rec: dict, strategy: str) -> dict | None:
    entry = rec["strategies"].get(strategy)
    return entry["analysis"] if entry else None


# ---------------------------------------------------------------- summary


def summarize(run: dict) -> dict:
    """Per-strategy recall/latency/slices with bootstrap CIs over dialogues."""
    cfg, records = run["config"], run["records"]
    strategies = cfg["strategies"]
    out = {
        "run_id": cfg.get("run_id"),
        "preset": cfg.get("preset"),
        "model": cfg.get("model"),
        "n_dialogues_total": len(cfg.get("specs", [])),
        "n_dialogues_done": len(records),
        "strategies": {},
    }
    for sname in strategies:
        pairs = [(rec, _analysis(rec, sname)) for rec in records]
        pairs = [(rec, a) for rec, a in pairs if a is not None]
        if not pairs:
            continue
        recalls = [a["recall_mean"] for _, a in pairs]
        lo, hi = bootstrap_ci(recalls)

        def _mean_of(fn):
            vals = [fn(a) for _, a in pairs]
            vals = [v for v in vals if v is not None]
            return round(mean(vals), 4) if vals else None

        by_kind = {k: _mean_of(lambda a, k=k: a.get("recall_by_kind", {}).get(k)) for k in KINDS}
        by_lag = {k: _mean_of(lambda a, k=k: a.get("recall_by_lag", {}).get(k)) for k in LAGS}
        slices: dict[str, dict] = {}
        for axis in ("topic", "persona", "scenario", "target_turns"):
            groups: dict[str, list[float]] = {}
            for rec, a in pairs:
                key = str(rec["spec"].get(axis))
                groups.setdefault(key, []).append(a["recall_mean"])
            slices[axis] = {k: round(mean(v), 4) for k, v in sorted(groups.items())}
        e2e = [
            median(rec["strategies"][sname]["e2e"]["per_rep_mean_s"])
            for rec, _ in pairs
            if rec["strategies"][sname].get("e2e", {}).get("per_rep_mean_s")
        ]
        out["strategies"][sname] = {
            "n_dialogues": len(pairs),
            "recall_mean": round(mean(recalls), 4),
            "recall_ci95": [lo, hi],
            "recall_direct": _mean_of(lambda a: a.get("recall_direct")),
            "recall_paraphrase": _mean_of(lambda a: a.get("recall_paraphrase")),
            "recall_by_kind": by_kind,
            "recall_by_lag": by_lag,
            "stale_rate": _mean_of(lambda a: a.get("stale_rate")),
            "tone_mean": _mean_of(lambda a: a.get("tone_mean")),
            "prompt_tokens": _mean_of(lambda a: a.get("prompt_tokens")),
            "cache_slots": _mean_of(lambda a: a.get("cache_slots")),
            "e2e_latency_median_s": round(median(e2e), 3) if e2e else None,
            "slices": slices,
        }
    return out


def compare_pairs(run: dict, pairs: list[tuple[str, str]]) -> dict[str, float]:
    """Paired-bootstrap p-values for strategy pairs on per-dialogue recall."""
    records = run["records"]
    pvals: dict[str, float] = {}
    for a_name, b_name in pairs:
        vals_a, vals_b = [], []
        for rec in records:
            aa, ab = _analysis(rec, a_name), _analysis(rec, b_name)
            if aa and ab:
                vals_a.append(aa["recall_mean"])
                vals_b.append(ab["recall_mean"])
        p = paired_bootstrap_p(vals_a, vals_b)
        if p is not None:
            pvals[f"{a_name} vs {b_name}"] = p
    return holm(pvals) if pvals else {}


# ---------------------------------------------------------------- rendering


def _fmt_ci(entry: dict) -> str:
    lo, hi = entry.get("recall_ci95", (None, None))
    if lo is None:
        return "—"
    return f"{entry['recall_mean']:.3f} [{lo:.3f}, {hi:.3f}]"


def render_interim_md(run_dir) -> str:
    """Render INTERIM.md for the current state of a run (progress + stats)."""
    run = load_run(run_dir)
    cfg, records = run["config"], run["records"]
    summary = summarize(run)
    total = summary["n_dialogues_total"]
    done = summary["n_dialogues_done"]
    walls = [rec.get("wall_s") or 0.0 for rec in records]
    wall_total = sum(walls)
    eta = (wall_total / done) * (total - done) if done else None

    lines = [
        f"# Промежуточный отчёт: {cfg.get('run_id')}",
        "",
        f"- Пресет: **{cfg.get('preset')}** · модель: `{cfg.get('model')}` · git SHA: `{cfg.get('git_sha', 'unknown')}`",
        f"- Прогресс: **{done}/{total}** батчей; GPU-время: {wall_total / 3600:.2f} ч"
        + (f"; оценка остатка: {eta / 3600:.2f} ч" if eta is not None else ""),
        f"- Повторов латентности на батч: {cfg.get('protocol', {}).get('latency_repeats', 0)} (+1 warmup)",
        "",
    ]

    # --- per-strategy table ---------------------------------------------------
    if summary["strategies"]:
        lines += [
            "## Качество и стоимость (по завершённым диалогам)",
            "",
            "| Стратегия | n | Recall [95 % ДИ] | direct | paraphr. | multihop | temporal | unansw. | e2e, с (мед.) | prompt токены |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for sname, s in summary["strategies"].items():
            bk = s["recall_by_kind"]

            def f3(v):
                return f"{v:.3f}" if v is not None else "—"

            lines.append(
                f"| {sname} | {s['n_dialogues']} | {_fmt_ci(s)} "
                f"| {f3(s['recall_direct'])} | {f3(s['recall_paraphrase'])} "
                f"| {f3(bk['multihop'])} | {f3(bk['temporal'])} | {f3(bk['unanswerable'])} "
                f"| {s['e2e_latency_median_s'] if s['e2e_latency_median_s'] is not None else '—'} "
                f"| {round(s['prompt_tokens']) if s['prompt_tokens'] is not None else '—'} |"
            )
        lines += ["", "## Recall по lag-зонам", "",
                  "| Стратегия | " + " | ".join(LAGS) + " |", "|---" * (len(LAGS) + 1) + "|"]
        for sname, s in summary["strategies"].items():
            row = " | ".join(
                f"{s['recall_by_lag'][k]:.3f}" if s["recall_by_lag"][k] is not None else "—" for k in LAGS
            )
            lines.append(f"| {sname} | {row} |")

        # --- slices -------------------------------------------------------------
        lines += ["", "## Средний recall по осям корпуса", ""]
        for axis, axis_name in (("topic", "Тема"), ("persona", "Персона"),
                                ("scenario", "Ситуация"), ("target_turns", "Длина")):
            keys = sorted({k for s in summary["strategies"].values() for k in s["slices"][axis]})
            if not keys:
                continue
            lines += [f"**{axis_name}:**", "",
                      "| Стратегия | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
            for sname, s in summary["strategies"].items():
                row = " | ".join(f"{s['slices'][axis].get(k, float('nan')):.3f}" for k in keys)
                lines.append(f"| {sname} | {row} |")
            lines.append("")

    # --- pairwise comparisons ---------------------------------------------------
    if done >= 5:
        names = list(summary["strategies"])
        key_pairs = []
        for ref in ("v4_rag_kv", "abl_rag_kv"):
            if ref in names:
                key_pairs += [(ref, other) for other in names if other != ref and other != "classic_kv_full"]
                if "classic_kv_full" in names:
                    key_pairs.append((ref, "classic_kv_full"))
        if not key_pairs and len(names) >= 2:
            key_pairs = [(names[0], n) for n in names[1:]]
        pvals = compare_pairs(run, key_pairs[:8])
        if pvals:
            lines += ["## Попарные сравнения recall (paired bootstrap, поправка Холма)", ""]
            for label, p in sorted(pvals.items()):
                mark = "**" if p < 0.05 else ""
                lines.append(f"- {label}: p = {mark}{p:.4f}{mark}")
            lines.append("")

    # --- anomalies -----------------------------------------------------------------
    reruns = [rec for rec in records if rec.get("reruns", 0) > 0]
    wall_median = median(walls) if walls else 0.0
    outliers = [rec for rec in records if wall_median and (rec.get("wall_s") or 0) > 2.5 * wall_median]
    lines += ["## Целостность", "",
              f"- батчей в манифесте: {len(records)}; перезапусков: {len(reruns)}"
              + (f" ({', '.join(r['batch_id'] for r in reruns[:5])})" if reruns else ""),
              f"- аномалии wall-time (> 2.5× медианы {wall_median / 60:.1f} мин): {len(outliers)}"
              + (f" ({', '.join(r['batch_id'] for r in outliers[:5])})" if outliers else ""),
              ""]
    return "\n".join(lines)

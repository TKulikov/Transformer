"""Summarize stress results into a markdown comparison (run after the CLI battery).

Focus: the author's objective — dialogue *tone* quality per unit of compute —
with fact recall/staleness as secondary columns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(v, spec="{:.2f}"):
    return "-" if v is None else spec.format(v)


def summarize(path: Path) -> str:
    r = json.load(path.open())
    lines = [f"\n# Стресс-отчёт: {r['model']} (device={r['device']})\n"]
    for session_name, sess in r["sessions"].items():
        lines.append(f"\n## {session_name} (ходов {sess['turns']}, проб {sess['n_probes']})\n")
        lines.append(
            "| стратегия | тон↑ | повтор.↓ | recall | early/mid/late | stale↓ | связность | "
            "prompt tok | embed tok | attn-слоты | max кэш | лат., с/ход | память S/M | банк add/up |"
        )
        lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for name, a in sess["strategies"].items():
            z = a["recall_by_zone"]
            ms = a.get("memory_stats")
            mem = f"{ms['stores']}/{ms['merges']}" if ms else "-"
            rs = a.get("rag_stats")
            bank = f"{rs['adds']}/{rs['upserts']}" if rs else "-"
            lines.append(
                f"| {name} | {fmt(a.get('tone_mean'))} | {fmt(a.get('rep_mean'))} | {fmt(a['recall_mean'])}"
                f" | {fmt(z.get('early'))} / {fmt(z.get('middle'))} / {fmt(z.get('late'))}"
                f" | {fmt(a['stale_rate'])} | {fmt(a['consistency_vs_reference'])}"
                f" | {a['prompt_tokens']} | {a.get('embed_tokens', 0)} | {a['cache_slots']} | {a['max_cache_len']}"
                f" | {a['mean_turn_latency_s']:.2f} | {mem} | {bank} |"
            )
        # Pareto hint: tone relative to gold, per unit of attention compute
        gold = sess["strategies"].get("classic_kv_full", {})
        gold_tone, gold_slots = gold.get("tone_mean"), gold.get("cache_slots")
        if gold_slots:
            lines.append("")
            lines.append("*Тон относительно эталона (full context) на 1/N attention-слотов:*")
            for name, a in sess["strategies"].items():
                tone, slots = a.get("tone_mean"), a["cache_slots"]
                if tone is None or not slots or not gold_tone:
                    continue
                rel = (tone / gold_tone) / (slots / gold_slots)
                lines.append(f"  - {name}: x{rel:.1f}")
    return "\n".join(lines)


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/stress")
    files = sorted(out_dir.glob("*_stress.json")) + sorted(out_dir.glob("*_battery100.json"))
    parts = [summarize(p) for p in files]
    report = "\n".join(parts)
    (out_dir / "STRESS_REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

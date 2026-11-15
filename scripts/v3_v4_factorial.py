"""Факторная абляция v3 vs v4: окно × центрирование × латентный K/V.

Заказ 15.11: v3_rag_centroid (w2, center, noKV) и v4_rag_kv (w5, center, KV)
различаются сразу несколькими ручками — чистого сравнения нет. Прогоняем
полную сетку 2×2×2 на лучшем слое (rag_layer=mid), battery100, и считаем
маргинальный эффект каждой ручки (окно / центрирование / K/V-канал).

Сетка: window {2,5} × rag_centered {0,1} × memory_kv {0,1}.
  w2_c1_kv0 = v3_rag_centroid; w5_c1_kv1 = v4_rag_kv;
  w5_c0_kv0 = rag_text_only.

Запуск: HF_HUB_OFFLINE=1 .venv/bin/python scripts/v3_v4_factorial.py
Выход:  results/v3_v4_factorial/{summary.json,REPORT.md}
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from tmem.dialogue import Strategy, run_session  # noqa: E402
from tmem.models import load_model  # noqa: E402
from tmem.stress import analyze_session, build_battery_100  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "v3_v4_factorial"
RAG_LAYER = "mid"
MAX_NEW = 24


def build_strategies() -> list[Strategy]:
    out = []
    for w, c, kv in itertools.product([2, 5], [False, True], [False, True]):
        name = f"w{w}_c{int(c)}_kv{int(kv)}"
        kw = dict(context="window", window=w, rag=True, rag_lambda=0.0,
                  rag_centered=c, rag_layer=RAG_LAYER)
        if kv:
            kw.update(memory_kv=True, memory_window_crit=0.5, memory_crit=0.95,
                      memory_merge_k=0.25, kv_slots=32)
        out.append(Strategy(name, **kw))
    return out


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"загрузка qwen35-4b на {device}, rag_layer={RAG_LAYER}...", flush=True)
    bundle = load_model("qwen35-4b", device=device, dtype=torch.bfloat16)
    session = build_battery_100()
    strategies = build_strategies()
    runs = {}
    recalls = {}
    for s in strategies:
        t0 = time.perf_counter()
        print(f"=== {s.name} ===", flush=True)
        runs[s.name] = run_session(bundle, session, s, max_new_tokens=MAX_NEW, seed=0)
        a = analyze_session({s.name: runs[s.name]})[s.name]
        recalls[s.name] = a["recall_mean"]
        print(f"  recall={a['recall_mean']} wall={time.perf_counter()-t0:.0f}s", flush=True)
        (DIR / "summary.json").write_text(
            json.dumps({"rag_layer": RAG_LAYER, "recalls": recalls}, indent=2), encoding="utf-8")
    write_report(recalls)
    print("готово:", DIR, flush=True)


def _marginal(recalls, knob_idx, val):
    # имя w{W}_c{C}_kv{K}; усредняем по двум другим ручкам
    vals = []
    for name, r in recalls.items():
        parts = name.replace("w", "").replace("_c", " ").replace("_kv", " ").split()
        w, c, kv = int(parts[0]), int(parts[1]), int(parts[2])
        bits = [w == 5, c == 1, kv == 1]  # window5, centered, kv
        if bits[knob_idx] == val:
            vals.append(r)
    return sum(vals) / len(vals) if vals else None


def write_report(recalls: dict) -> None:
    lines = [
        "# Факторная абляция v3 vs v4 (battery100, rag_layer=mid)",
        "",
        "Сетка: окно {2,5} × центрирование {0,1} × латентный K/V {0,1}.",
        "w2_c1_kv0 = v3_rag_centroid; w5_c1_kv1 = v4_rag_kv; w5_c0_kv0 = rag_text_only.",
        "",
        "| window | centered | KV | recall |",
        "|---|---|---|---|",
    ]
    for name, r in recalls.items():
        parts = name.replace("w", "").replace("_c", " ").replace("_kv", " ").split()
        lines.append(f"| {parts[0]} | {parts[1]} | {parts[2]} | {r:.3f} |")
    lines += ["", "## Маргинальные эффекты (среднее по двум другим ручкам)", "",
              "| Ручка | off → on | Δ |", "|---|---|---|"]
    for idx, label in [(0, "окно 2→5"), (1, "центрирование"), (2, "латентный K/V")]:
        off, on = _marginal(recalls, idx, False), _marginal(recalls, idx, True)
        delta = f"{on-off:+.3f}" if (off is not None and on is not None) else "—"
        lines.append(f"| {label} | {off:.3f} → {on:.3f} | {delta} |")
    lines += ["", "## v3 vs v4 (как в батарее)", "",
              f"- v3 (w2_c1_kv0): {recalls.get('w2_c1_kv0')}",
              f"- v4 (w5_c1_kv1): {recalls.get('w5_c1_kv1')}",
              f"- rag_text_only (w5_c0_kv0): {recalls.get('w5_c0_kv0')}", ""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

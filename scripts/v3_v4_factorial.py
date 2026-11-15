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
    metrics = {}
    for s in strategies:
        t0 = time.perf_counter()
        print(f"=== {s.name} ===", flush=True)
        runs[s.name] = run_session(bundle, session, s, max_new_tokens=MAX_NEW, seed=0)
        a = analyze_session({s.name: runs[s.name]})[s.name]
        metrics[s.name] = {"recall": a["recall_mean"], "tone": a.get("tone_mean"),
                           "rep": a["rep_mean"]}
        print(f"  recall={a['recall_mean']} tone={a.get('tone_mean')} "
              f"rep={a['rep_mean']} wall={time.perf_counter()-t0:.0f}s", flush=True)
        (DIR / "summary.json").write_text(
            json.dumps({"rag_layer": RAG_LAYER, "metrics": metrics}, indent=2), encoding="utf-8")
    write_report(metrics)
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


def _flat(metrics, key):
    return {n: m[key] for n, m in metrics.items() if m.get(key) is not None}


def write_report(metrics: dict) -> None:
    lines = [
        "# Факторная абляция v3 vs v4 (battery100, rag_layer=mid)",
        "",
        "Сетка: окно {2,5} × центрирование {0,1} × латентный K/V {0,1}.",
        "w2_c1_kv0 = v3; w5_c1_kv1 = v4; w5_c0_kv0 = rag_text_only.",
        "**tone** — нить/тема диалога (косинус ответа к банку); целевая метрика K/V-канала.",
        "**rep** — деградация (повторы). recall — факты (работа RAG, не K/V).",
        "",
        "| window | centered | KV | recall | tone | rep |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in metrics.items():
        parts = name.replace("w", "").replace("_c", " ").replace("_kv", " ").split()
        tone = f"{m['tone']:.3f}" if m.get("tone") is not None else "—"
        lines.append(f"| {parts[0]} | {parts[1]} | {parts[2]} | {m['recall']:.3f} | {tone} | {m['rep']:.3f} |")
    for key in ("tone", "recall"):
        flat = _flat(metrics, key)
        lines += ["", f"## Маргинальные эффекты по {key} (среднее по двум другим ручкам)", "",
                  "| Ручка | off → on | Δ |", "|---|---|---|"]
        for idx, label in [(0, "окно 2→5"), (1, "центрирование"), (2, "латентный K/V")]:
            off, on = _marginal(flat, idx, False), _marginal(flat, idx, True)
            delta = f"{on-off:+.3f}" if (off is not None and on is not None) else "—"
            off_s = f"{off:.3f}" if off is not None else "—"
            on_s = f"{on:.3f}" if on is not None else "—"
            lines.append(f"| {label} | {off_s} → {on_s} | {delta} |")
    lines += ["", "## v3 vs v4 (recall / tone)", ""]
    for nm, label in [("w2_c1_kv0", "v3"), ("w5_c1_kv1", "v4"), ("w5_c0_kv0", "rag_text_only"),
                      ("w5_c1_kv0", "best-recall (w5,c1,noKV)")]:
        m = metrics.get(nm, {})
        tone = f"{m['tone']:.3f}" if m.get("tone") is not None else "—"
        lines.append(f"- {label} ({nm}): recall {m.get('recall')} / tone {tone}")
    lines += [""]
    (DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

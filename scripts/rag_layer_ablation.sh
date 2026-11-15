#!/usr/bin/env bash
# Абляция слоя RAG-эмбеддинга: battery100, last (hidden_state[-1]) vs mid (латентный).
# Заказ 15.11: прогнать диалоговый тест после смены RAG с финального слоя на латентный ключ.
# GPU последовательно: сначала last (базовый), затем mid.
set -u
cd /home/morgot/Documents/Transformer
STRATS="classic_kv_full rag_text_only v3_rag_centroid v4_rag_kv"
for LAYER in last mid; do
  echo "[$(date '+%F %T')] === rag-layer=$LAYER ==="
  HF_HUB_OFFLINE=1 .venv/bin/python -m tmem.cli --model qwen35-4b --device cuda --dtype bfloat16 \
    --battery100 --strategies $STRATS --rag-layer "$LAYER" \
    --out "results/rag_layer_ablation/$LAYER"
  echo "[$(date '+%F %T')] rag-layer=$LAYER exit=$?"
done
.venv/bin/python - <<'EOF'
import json, pathlib
base = pathlib.Path("results/rag_layer_ablation")
data = {}
for layer in ("last", "mid"):
    p = base / layer / "qwen35-4b_battery100.json"
    if p.exists():
        data[layer] = json.loads(p.read_text())
rows = {}
for layer, d in data.items():
    for sess, sd in d.get("sessions", {}).items():
        for strat, a in sd.get("strategies", {}).items():
            rows.setdefault(strat, {})[layer] = a.get("recall_mean")
lines = ["# Абляция слоя RAG-эмбеддинга (battery100, Qwen3.5-4B bf16)", "",
         "last = hidden_state[-1] (базовый RAG); mid = латентный слой смыслов.", "",
         "| Стратегия | recall last | recall mid | Δ (mid−last) |", "|---|---|---|---|"]
for strat, r in rows.items():
    la, mi = r.get("last"), r.get("mid")
    delta = f"{mi-la:+.3f}" if (la is not None and mi is not None) else "—"
    lines.append(f"| {strat} | {la if la is not None else '—'} | {mi if mi is not None else '—'} | {delta} |")
lines += ["", "classic_kv_full — контроль (не использует RAG, должен быть стабилен).", ""]
(base / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
EOF
echo "[$(date '+%F %T')] ablation complete"

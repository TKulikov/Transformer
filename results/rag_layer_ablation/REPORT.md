# Абляция слоя RAG-эмбеддинга (battery100, Qwen3.5-4B bf16)

last = hidden_state[-1] (базовый RAG); mid = латентный слой смыслов.

| Стратегия | recall last | recall mid | Δ (mid−last) |
|---|---|---|---|
| classic_kv_full | 1.0 | 1.0 | +0.000 |
| rag_text_only | 0.625 | 0.792 | +0.167 |
| v3_rag_centroid | 0.708 | 0.833 | +0.125 |
| v4_rag_kv | 0.583 | 0.792 | +0.209 |

classic_kv_full — контроль (не использует RAG, должен быть стабилен).

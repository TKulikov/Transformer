# Живая память как ядро: TTL-динамика и gist-readout

Тезис: ядро — живая TTL-память темы (refresh/decay/eviction), gist — один readout.
decay on/off = живая vs статическая; alpha/lambda = сила gist (0 = без gist).

## Часть A — код (taskq_large, semtree mean/tau=0.6)

| Условие | Reuse /10 | Hits /10 | Def hits /10 | Слотов | gist_used |
|---|---|---|---|---|---|
| nobank | 9 | 7 | 4 | 0 | 0 |
| static_a05 | 9 | 7 | 4 | 512 | 5 |
| living_a05 | 9 | 7 | 4 | 512 | 5 |
| static_a10 | 9 | 7 | 4 | 512 | 5 |
| living_a10 | 9 | 6 | 4 | 512 | 5 |

## Часть B — диалог (thread-тест, дрейф темы)

| Условие | recall | confusion (потеря нити) | tone | K/V inj |
|---|---|---|---|---|
| v3_nomem | 0.786 | 0.000 | 0.861 | 0 |
| kv_living_a0 | 0.833 | 0.143 | 0.859 | 249539 |
| kv_static_a0 | 0.786 | 0.143 | 0.856 | 249539 |
| kv_living_a05 | 0.738 | 0.286 | 0.859 | 249539 |
| kv_living_a10 | 0.500 | 0.429 | 0.861 | 247491 |

Часть A: банк влияет на поиск через gist (alpha=0 = базлайн без банка);
living vs static при alpha>0 изолирует TTL-динамику. Часть B: K/V-банк входит
в генерацию напрямую, lambda=0 = живая память БЕЗ gist; kv_static_a0 = статический банк.
Ниже confusion = лучше удержание нити. Данные: `code_<cond>.json`, `dlg_<cond>.json`.

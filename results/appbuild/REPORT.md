# App-building: живая память помнит СВОЙ код (reuse-vs-write)

Окно=1, 5 шагов. own_reuse — переиспользование СВОЕГО раннего кода.

| Условие | reuse (всего) | own_reuse (свой код) | valid |
|---|---|---|---|
| no_memory | 8/8 | 4/4 | 0/5 |
| static | 8/8 | 4/4 | 0/5 |
| living | 8/8 | 4/4 | 0/5 |

## По шагам (own_reuse)

| Условие | rate_limiter | cache_result | guarded_call | batch_guarded | invoice_guarded |
|---|---|---|---|---|---|
| no_memory | — | — | rate_limiter=да, cache_result=да | guarded_call=да | guarded_call=да |
| static | — | — | rate_limiter=да, cache_result=да | guarded_call=да | guarded_call=да |
| living | — | — | rate_limiter=да, cache_result=да | guarded_call=да | guarded_call=да |

no_memory — только окно; static — ChunkBank-поиск; living — ChunkBank+TTL-gist.
Гипотеза: living лучше помнит свой ранний код (own_reuse) при коротком окне. Данные: `<cond>.json`.

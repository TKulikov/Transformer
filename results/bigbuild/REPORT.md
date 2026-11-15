# Большая сборка: живая память на масштабе (36 шагов)

window=1, k=5. reuse=вызов prior-функции; rewrite=переопределение; miss=забыл.
Гипотеза: static деградирует с ростом базы, living_memctx удерживает направление.

| Условие | reuse | rewrite | miss | core | service | pipeline | банк |
|---|---|---|---|---|---|---|---|
| static | 2/60 | 0 | 58 | 0/0 | 2/24 | 0/36 | 41 |
| living_memctx | 6/60 | 0 | 54 | 0/0 | 4/24 | 2/36 | 36 |

## Кривая cumulative reuse (по шагам с reuse-целями)

| шаг | фаза | static cum | living cum |
|---|---|---|---|
| 12 ingest_records | service | 0/2 | 1/2 |
| 13 clean_records | service | 0/4 | 1/4 |
| 14 enrich_records | service | 0/6 | 1/6 |
| 15 resilient_call | service | 0/8 | 1/8 |
| 16 audit_trail | service | 0/10 | 1/10 |
| 17 name_records | service | 0/12 | 2/12 |
| 18 batch_records | service | 2/14 | 3/14 |
| 19 time_series | service | 2/16 | 3/16 |
| 20 money_report | service | 2/18 | 3/18 |
| 21 safe_transform | service | 2/20 | 3/20 |
| 22 cached_ingest | service | 2/22 | 3/22 |
| 23 logged_batch | service | 2/24 | 4/24 |
| 24 etl_pipeline | pipeline | 2/27 | 4/27 |
| 25 report_pipeline | pipeline | 2/30 | 4/30 |
| 26 resilient_etl | pipeline | 2/33 | 4/33 |
| 27 audit_pipeline | pipeline | 2/36 | 4/36 |
| 28 naming_pipeline | pipeline | 2/39 | 4/39 |
| 29 timeseries_pipeline | pipeline | 2/42 | 4/42 |
| 30 safe_etl | pipeline | 2/45 | 5/45 |
| 31 cached_pipeline | pipeline | 2/48 | 5/48 |
| 32 logged_pipeline | pipeline | 2/51 | 6/51 |
| 33 money_pipeline | pipeline | 2/54 | 6/54 |
| 34 full_pipeline | pipeline | 2/57 | 6/57 |
| 35 resilient_full | pipeline | 2/60 | 6/60 |

static — обычные эмбеддинги чанков; living_memctx — контекстуализация K/V-инжектом
живой памяти при кодировании + TTL-эволюция + gist. Данные: `<cond>.json`.

# Семантическое дерево связей (taskq_large): группы по смыслу, не по файлам

Онлайн leader-кластеризация в deviation-пространстве; ключ листа = отклонение
от среднего группы; поиск: top-3 группы по центроиду → residual-поиск внутри.
k=6 заметок. Сходства deviation (percentiles): {'mid': {'p50': -0.074, 'p90': 0.34, 'p99': 0.984}, 'mean': {'p50': -0.066, 'p90': 0.355, 'p99': 1.0}}.
Ингестия (с): {'semtree': 43.1, 'flat_legacy': 62.4}.

| Условие | Групп | Одиночек | Макс группа | Reuse /10 | Hits /10 | Def hits /10 |
|---|---|---|---|---|---|---|
| semtree_mid_t40 | 17 | 1 | 54 | 8 | 8 | 5 |
| semtree_mid_t60 | 30 | 8 | 36 | 7 | 9 | 4 |
| semtree_mid_t80 | 35 | 14 | 36 | 7 | 9 | 4 |
| semtree_mean_t40 | 18 | 1 | 54 | 7 | 6 | 4 |
| semtree_mean_t60 | 30 | 8 | 36 | 9 | 7 | 4 |
| semtree_mean_t80 | 44 | 22 | 18 | 8 | 7 | 4 |
| flat_legacy | — | — | — | 8 | 5 | 1 |

## Выбранные группы (stage 1, top-3 имени)

| Условие | retry_backoff | parse_clamp | batch_dedupe | slug_total | invoice_summary |
|---|---|---|---|---|---|
| semtree_mid_t40 | [36] process/reprocess_failed; [9] clamp/parse_duration | [9] clamp/parse_duration; [18] timeout_ms/timeout_ms | [9] clamp/parse_duration; [36] process/reprocess_failed | [9] clamp/parse_duration; [18] summarize/summarize | [18] summary/summary; [36] process/reprocess_failed |
| semtree_mid_t60 | [18] reprocess_failed/reprocess_failed; [18] process/process | [18] validate/validate; [18] timeout_ms/timeout_ms | [1] chunked; [18] process/process | [1] compute_total; [1] slugify | [18] summary/summary; [18] process/process |
| semtree_mid_t80 | [18] reprocess_failed/reprocess_failed; [18] process/process | [1] backoff_delay; [18] validate/validate | [1] chunked; [18] process/process | [1] compute_total; [1] backoff_delay | [18] summary/summary; [18] process/process |
| semtree_mean_t40 | [36] process/reprocess_failed; [24] now_ms/parse_duration | [24] now_ms/parse_duration; [3] clamp/chunked | [3] clamp/chunked; [36] process/reprocess_failed | [19] compute_total/summarize; [3] clamp/chunked | [18] summary/summary; [1] __init__ |
| semtree_mean_t60 | [20] backoff_delay/retry; [18] process/process | [18] timeout_ms/timeout_ms; [1] clamp | [1] chunked; [18] process/process | [1] compute_total; [18] summarize/summarize | [18] summary/summary; [1] __init__ |
| semtree_mean_t80 | [18] reprocess_failed/reprocess_failed; [18] process/process | [18] timeout_ms/timeout_ms; [1] clamp | [1] chunked; [18] process/process | [1] compute_total; [18] summarize/summarize | [18] summary/summary; [1] __init__ |
| flat_legacy | — | — | — | — | — |

Def hits — определение хелпера в notes (не только вызов); flat_legacy — референс.
Данные: `summary.json`, `<cond>.json`.

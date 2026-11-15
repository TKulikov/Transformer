# Память на большом мультифайловом проекте taskq_large

Дерево: {'file': 83, 'class': 75, 'function': 392, 'usage': 110}; ингестия 56.2 с. Контекст — описание проекта без тел хелперов.
Retrieval hit — имя в notes как def ИЛИ вызов. Макс reuse/hits = 10 (5 проб × 2).

| Режим | Задача | Reuse | Retrieval hit |
|---|---|---|---|
| no_memory | retry_backoff | retry=да, backoff_delay=нет | retry=нет, backoff_delay=нет |
| no_memory | parse_clamp | parse_duration=нет, clamp=да | parse_duration=нет, clamp=нет |
| no_memory | batch_dedupe | dedupe=нет, chunked=нет | dedupe=нет, chunked=нет |
| no_memory | slug_total | slugify=да, compute_total=нет | slugify=нет, compute_total=нет |
| no_memory | invoice_summary | summarize=нет, domain_metric=да | summarize=нет, domain_metric=нет |
| flat | retry_backoff | retry=нет, backoff_delay=да | retry=нет, backoff_delay=да |
| flat | parse_clamp | parse_duration=да, clamp=да | parse_duration=да, clamp=да |
| flat | batch_dedupe | dedupe=нет, chunked=нет | dedupe=нет, chunked=нет |
| flat | slug_total | slugify=да, compute_total=да | slugify=нет, compute_total=да |
| flat | invoice_summary | summarize=да, domain_metric=да | summarize=да, domain_metric=да |
| descent | retry_backoff | retry=да, backoff_delay=нет | retry=нет, backoff_delay=нет |
| descent | parse_clamp | parse_duration=да, clamp=да | parse_duration=да, clamp=нет |
| descent | batch_dedupe | dedupe=нет, chunked=нет | dedupe=нет, chunked=нет |
| descent | slug_total | slugify=да, compute_total=нет | slugify=нет, compute_total=нет |
| descent | invoice_summary | summarize=да, domain_metric=нет | summarize=да, domain_metric=нет |
| union | retry_backoff | retry=нет, backoff_delay=да | retry=нет, backoff_delay=да |
| union | parse_clamp | parse_duration=да, clamp=да | parse_duration=да, clamp=да |
| union | batch_dedupe | dedupe=нет, chunked=нет | dedupe=нет, chunked=нет |
| union | slug_total | slugify=да, compute_total=да | slugify=нет, compute_total=да |
| union | invoice_summary | summarize=да, domain_metric=нет | summarize=да, domain_metric=да |
| shallow | retry_backoff | retry=нет, backoff_delay=да | retry=нет, backoff_delay=да |
| shallow | parse_clamp | parse_duration=да, clamp=да | parse_duration=да, clamp=да |
| shallow | batch_dedupe | dedupe=нет, chunked=нет | dedupe=нет, chunked=нет |
| shallow | slug_total | slugify=да, compute_total=да | slugify=нет, compute_total=да |
| shallow | invoice_summary | summarize=да, domain_metric=да | summarize=да, domain_metric=да |

## Сводка (сумма по пробам, макс 10)

- no_memory: reuse 4/10, retrieval hits 0/10
- flat: reuse 7/10, retrieval hits 6/10
- descent: reuse 5/10, retrieval hits 2/10
- union: reuse 6/10, retrieval hits 6/10
- shallow: reuse 7/10, retrieval hits 6/10

Данные: `summary.json`, `<mode>.json`.

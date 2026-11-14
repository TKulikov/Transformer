# Память на реалистичном проекте taskq (клиент-сервер)

Дерево: {'file': 21, 'class': 25, 'function': 79, 'usage': 33}; ингестия 12.5 с. Контекст — описание проекта без тел хелперов.
Retrieval hit — имя в notes как def ИЛИ вызов (исправленная метрика).

| Режим | Задача | Reuse | Retrieval hit |
|---|---|---|---|
| no_memory | retry_submit | retry=да, backoff_delay=нет | retry=нет, backoff_delay=нет |
| no_memory | parse_clamp | parse_duration=нет, clamp=да | parse_duration=нет, clamp=нет |
| no_memory | batch_dedupe | chunked=нет, dedupe=нет | chunked=нет, dedupe=нет |
| no_memory | snapshot_metrics | snapshot=да, summary=да | snapshot=нет, summary=нет |
| flat | retry_submit | retry=да, backoff_delay=нет | retry=да, backoff_delay=нет |
| flat | parse_clamp | parse_duration=да, clamp=да | parse_duration=да, clamp=нет |
| flat | batch_dedupe | chunked=да, dedupe=нет | chunked=да, dedupe=да |
| flat | snapshot_metrics | snapshot=да, summary=нет | snapshot=нет, summary=нет |
| descent | retry_submit | retry=да, backoff_delay=нет | retry=нет, backoff_delay=нет |
| descent | parse_clamp | parse_duration=нет, clamp=да | parse_duration=нет, clamp=нет |
| descent | batch_dedupe | chunked=нет, dedupe=нет | chunked=нет, dedupe=нет |
| descent | snapshot_metrics | snapshot=да, summary=да | snapshot=нет, summary=нет |
| union | retry_submit | retry=да, backoff_delay=нет | retry=да, backoff_delay=нет |
| union | parse_clamp | parse_duration=нет, clamp=да | parse_duration=нет, clamp=нет |
| union | batch_dedupe | chunked=нет, dedupe=нет | chunked=да, dedupe=да |
| union | snapshot_metrics | snapshot=да, summary=да | snapshot=нет, summary=нет |
| shallow | retry_submit | retry=да, backoff_delay=нет | retry=да, backoff_delay=нет |
| shallow | parse_clamp | parse_duration=нет, clamp=да | parse_duration=нет, clamp=нет |
| shallow | batch_dedupe | chunked=да, dedupe=нет | chunked=да, dedupe=да |
| shallow | snapshot_metrics | snapshot=да, summary=нет | snapshot=нет, summary=нет |

## Сводка reuse (сумма по пробам, макс 8)

- no_memory: reuse 4/8, retrieval hits 0/8
- flat: reuse 5/8, retrieval hits 4/8
- descent: reuse 4/8, retrieval hits 0/8
- union: reuse 4/8, retrieval hits 3/8
- shallow: reuse 4/8, retrieval hits 3/8

Данные: `summary.json`, `<mode>.json`.

# Ретест поисков по коду (taskq_large): латент vs hidden[-1], нужен ли K/V-инжект

Проект: 82 файлов, 392 функций. 5 задач × 2 хелпера = 10 бинарных исходов.
Ингестия (с): {'flat_last_s': 64.3, 'flat_mid_s': 61.8, 'flat_legacy_s': 62.0, 'latent_bank_s': 39.9, 'hier_s': 71.6}.
reuse — имя хелпера в ответе; retrieval hit — def/вызов в notes; valid — синтаксис.

| Условие | Reuse /10 | Retrieval hits /10 | Valid /5 | Время, с |
|---|---|---|---|---|
| no_memory | 4 | 0 | 0 | 31.4 |
| flat_last | 5 | 3 | 0 | 22.0 |
| flat_mid | 4 | 4 | 0 | 21.1 |
| flat_legacy | 8 | 5 | 1 | 19.1 |
| latent_bank | 6 | 5 | 0 | 22.7 |
| hier_flat | 7 | 6 | 0 | 16.6 |
| kv_only | 4 | 0 | 0 | 50.6 |
| hier_flat_kv | 6 | 6 | 0 | 42.4 |

## По задачам (reuse)

| Условие | retry_backoff | parse_clamp | batch_dedupe | slug_total | invoice_summary |
|---|---|---|---|---|---|
| no_memory | retry=да, backoff_delay=нет | parse_duration=нет, clamp=да | dedupe=нет, chunked=нет | slugify=да, compute_total=нет | summarize=нет, domain_metric=да |
| flat_last | retry=нет, backoff_delay=да | parse_duration=да, clamp=да | dedupe=нет, chunked=нет | slugify=да, compute_total=нет | summarize=да, domain_metric=нет |
| flat_mid | retry=нет, backoff_delay=нет | parse_duration=нет, clamp=да | dedupe=нет, chunked=нет | slugify=да, compute_total=да | summarize=да, domain_metric=нет |
| flat_legacy | retry=да, backoff_delay=нет | parse_duration=да, clamp=да | dedupe=да, chunked=нет | slugify=да, compute_total=да | summarize=да, domain_metric=да |
| latent_bank | retry=нет, backoff_delay=да | parse_duration=да, clamp=да | dedupe=нет, chunked=нет | slugify=да, compute_total=да | summarize=нет, domain_metric=да |
| hier_flat | retry=нет, backoff_delay=да | parse_duration=да, clamp=да | dedupe=нет, chunked=нет | slugify=да, compute_total=да | summarize=да, domain_metric=да |
| kv_only | retry=нет, backoff_delay=нет | parse_duration=нет, clamp=да | dedupe=нет, chunked=нет | slugify=да, compute_total=нет | summarize=да, domain_metric=да |
| hier_flat_kv | retry=нет, backoff_delay=да | parse_duration=да, clamp=да | dedupe=нет, chunked=нет | slugify=да, compute_total=да | summarize=да, domain_metric=нет |

## K/V-инъекция

- kv_only: {'injected_slots': 92032, 'steps': 2876, 'slots_left': 157}
- hier_flat_kv: {'injected_slots': 64384, 'steps': 2012, 'slots_left': 259}

flat_last/flat_mid/flat_legacy — плоский ChunkBank (ключ=активации блока, centered);
latent_bank — LatentRagBank (ключ=norm(q_сигнатура+β·тело), mid, TTL-gist запрос);
hier_flat — HierLatentBank mode=flat; kv_only — только K/V-инъекция (слот=mid-mean функции);
hier_flat_kv — hier_flat + K/V-инъекция. Данные: `summary.json`, `<cond>.json`.

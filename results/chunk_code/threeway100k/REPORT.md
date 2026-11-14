# Три варианта памяти на файле ~100k токенов

beta=1.0, alpha=0.5; хвост: {'line_start': 9484, 'dropped_lines': 11}.
Ингестия: {'flat_s': 163.1, 'hier_s': 198.5, 'latent_s': 148.6, 'latent_stats': {'adds': 1218, 'evictions': 0, 'gist_used': 4}}.

| Условие | Задача | Валид | Reuse | Retrieval hit |
|---|---|---|---|---|
| old_flat | continue | нет | — | — |
| old_flat | named_early | нет | normalize_records=да, compute_total=да | normalize_records=нет, compute_total=нет |
| old_flat | clean_mid | нет | group_by=нет, moving_average=да | group_by=нет, moving_average=нет |
| old_flat | clean_total | да | filter_active=нет, compute_total=да | filter_active=нет, compute_total=нет |
| hier | continue | да | — | — |
| hier | named_early | нет | normalize_records=да, compute_total=да | normalize_records=нет, compute_total=нет |
| hier | clean_mid | да | group_by=нет, moving_average=нет | group_by=нет, moving_average=нет |
| hier | clean_total | да | filter_active=нет, compute_total=да | filter_active=нет, compute_total=нет |
| latent | continue | да | — | — |
| latent | named_early | да | normalize_records=да, compute_total=да | normalize_records=нет, compute_total=нет |
| latent | clean_mid | да | group_by=нет, moving_average=нет | group_by=нет, moving_average=нет |
| latent | clean_total | да | filter_active=да, compute_total=да | filter_active=нет, compute_total=нет |

Данные: `summary.json`, `<condition>.json`.

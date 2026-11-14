# Гибридная память для кода: дерево + латентный слой + TTL-gist

beta=1.0, alpha=0.5; ингестия {'wall_s': 164.0, 'stats': {'ingested': 1603, 'declarations': 1218, 'usages': 298, 'gist_used': 4}}.

| Задача | Валид | Reuse | Retrieval hit | Ветки |
|---|---|---|---|---|
| continue | нет | — | — | 3 |
| named_early | да | normalize_records=да, compute_total=да | normalize_records=нет, compute_total=нет | 3 |
| clean_mid | нет | group_by=нет, moving_average=нет | group_by=нет, moving_average=нет | 3 |
| clean_total | да | filter_active=да, compute_total=да | filter_active=нет, compute_total=нет | 3 |

## Сравнение с тремя вариантами (reuse на чистых пробах)

| Вариант | clean_mid | clean_total |
|---|---|---|
| old_flat | 1/2 | 1/2 |
| hier | 0/2 | 1/2 |
| latent | 0/2 | 2/2 |
| **hier+latent** | 0/2 | 2/2 |

Данные: `summary.json` (ответы, injected notes).

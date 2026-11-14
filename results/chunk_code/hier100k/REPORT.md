# Иерархическая память на файле ~100k токенов

Дерево файл → класс → метод → вызовы; ключ узла = среднее активаций блока, центрирование отклонением от родителя (метод от класса, вызов от объявления); поиск: объявления по абсолютным ключам → спуск к использованиям по residual-ключам.

- Дерево: 1603 узлов {'file': 1, 'class': 86, 'function': 1218, 'usage': 298}.
- Ингестия: hier 200.0 с, flat 162.6 с.
- Хвост промпта: {'line_start': -1, 'helper_mentions': 7, 'masked': True} (имена хелперов не leaking).

| Условие | Задача | Валид | Reuse | Retrieval hit | Ветки |
|---|---|---|---|---|---|
| no_memory | continue | да | — | — | — |
| no_memory | named_early | да | normalize_records=да, compute_total=да | normalize_records=нет, compute_total=нет | — |
| no_memory | clean_mid | да | group_by=нет, moving_average=нет | group_by=нет, moving_average=нет | — |
| no_memory | clean_total | да | filter_active=нет, compute_total=нет | filter_active=нет, compute_total=нет | — |
| flat_ck | continue | да | — | — | — |
| flat_ck | named_early | да | normalize_records=да, compute_total=да | normalize_records=нет, compute_total=нет | — |
| flat_ck | clean_mid | да | group_by=нет, moving_average=нет | group_by=нет, moving_average=нет | — |
| flat_ck | clean_total | да | filter_active=нет, compute_total=нет | filter_active=нет, compute_total=нет | — |
| hier | continue | да | — | — | 2 |
| hier | named_early | да | normalize_records=да, compute_total=да | normalize_records=нет, compute_total=нет | 1 |
| hier | clean_mid | да | group_by=да, moving_average=нет | group_by=нет, moving_average=нет | 1 |
| hier | clean_total | да | filter_active=нет, compute_total=нет | filter_active=нет, compute_total=нет | 1 |

Данные: `summary.json`, `<condition>.json` (ответы, injected notes).

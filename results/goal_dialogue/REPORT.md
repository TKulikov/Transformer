# Goal-directedness: живая память удерживает ЦЕЛЬ (рекуррентную)

Диалог: 31 ходов, 5 проб. Цель: запуск MVP к пятнице (рекуррентна).
Тангенты: Rust / маркетинг / логотип. goal_recall — цель в ответе; goal_confusion — тангент вместо цели.
Гипотеза: merge подкрепляет рекуррентную цель → living_merge лучше удерживает направление.

| Условие | goal_recall | goal_confusion | tone |
|---|---|---|---|
| v3_clean | 0.800 | 0.400 | 0.887 |
| living_merge | 0.800 | 0.600 | 0.891 |
| living_nomerge | 0.533 | 0.200 | 0.891 |
| static_merge | 0.800 | 0.600 | 0.891 |

v3_clean — без живой памяти; living_merge — merge_k=0.25+TTL; living_nomerge — v1 mean+TTL;
static_merge — merge без TTL. Ниже goal_confusion = лучше удержание цели. Данные: `<cond>.json`.

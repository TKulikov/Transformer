# ЧЕСТНЫЙ app-building: живая память помнит СВОЙ код (без утечки, AST-скоринг)

10 шагов, window=1, сиды [0, 1, 2].
reuse = вызов prior-функции; rewrite = переопределение; miss = отсутствие.

| Условие | reuse (mean) | rewrite | miss | всего |
|---|---|---|---|---|
| clean_rag | 7.0 | 0.0 | 2.0 | 9 |
| static_chunks | 9.0 | 0.0 | 0.0 | 9 |
| living_full | 9.0 | 0.0 | 0.0 | 9 |

## По сидам (reuse)

| Условие | seed0 | seed1 | seed2 |
|---|---|---|---|
| clean_rag | 7 | 7 | 7 |
| static_chunks | 9 | 9 | 9 |
| living_full | 9 | 9 | 9 |

Гипотеза: living_full > static_chunks > clean_rag по reuse своего кода.
clean_rag — окно без памяти кода; static_chunks — статический поиск; living_full — вся живая система.
Данные: `<cond>_s<seed>.json`.

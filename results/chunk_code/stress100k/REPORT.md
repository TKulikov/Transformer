# Стресс-тест чанковой памяти: файл ~100k токенов

## Формулировка вариации (со слов заказчика)

«В ходе генерации чанка мы усредняем активации латентного банка, используем финальное усреднение как embedding, а как значение — контекстуализированный чанк».

Соответствие коду: `embed_block_key(layers="mean")` — усреднение пулов скрытых состояний по токенам каждого слоя и по слоям (ключ/embedding); `embed_chunk(mode="span")` — пул по токенам чанка в контексте соседнего кода (значение); оба центрируются бегущим средним (`ChunkBank`).

## Файл и разбиение на функции

- Файл: 99338 токенов, 1218 функций (ast), валидный Python.
- Чанков: 1305; функций, целиком попавших в один чанк: 1218/1218 (100.0%).
- Размеры чанков (токены): min 13, медиана 91, max 180.

## Ингестия в банк

- 1305 чанков за 164.4 с; 390893 embed-токенов; банк: {'adds': 1305, 'upserts': 0, 'evictions': 0}.

## Пробы «дописать» (хвост файла в промпте, 192 новых токена, greedy)

| Условие | Задача | Валидный код | Reuse | Retrieval hit |
|---|---|---|---|---|
| no_memory | continue | да | — | — |
| no_memory | named_early | да | normalize_records=да, compute_total=да | normalize_records=нет, compute_total=нет |
| no_memory | clean_mid | да | group_by=нет, moving_average=нет | group_by=нет, moving_average=нет |
| no_memory | clean_total | да | filter_active=да, compute_total=да | filter_active=нет, compute_total=нет |
| ck_key | continue | нет | — | — |
| ck_key | named_early | нет | normalize_records=да, compute_total=да | normalize_records=нет, compute_total=нет |
| ck_key | clean_mid | да | group_by=нет, moving_average=нет | group_by=нет, moving_average=нет |
| ck_key | clean_total | да | filter_active=да, compute_total=да | filter_active=нет, compute_total=нет |
| ck_value | continue | да | — | — |
| ck_value | named_early | да | normalize_records=да, compute_total=да | normalize_records=нет, compute_total=нет |
| ck_value | clean_mid | да | group_by=нет, moving_average=нет | group_by=нет, moving_average=нет |
| ck_value | clean_total | да | filter_active=да, compute_total=да | filter_active=нет, compute_total=нет |
| ck_maxsim | continue | да | — | — |
| ck_maxsim | named_early | да | normalize_records=да, compute_total=нет | normalize_records=нет, compute_total=нет |
| ck_maxsim | clean_mid | нет | group_by=нет, moving_average=нет | group_by=нет, moving_average=нет |
| ck_maxsim | clean_total | да | filter_active=да, compute_total=да | filter_active=нет, compute_total=нет |

continue — свободное продолжение файла (helper_calls в JSON); named_early — имена хелперов в вопросе; clean_mid/clean_total — имена НЕ названы, только описание (чистые пробы).

## Находки

1. Разбиение на функции работает идеально: 1218/1218 функций (100 %) целиком попали в один чанк; 1305 чанка (функции + заголовки классов), медиана 91 токен. Прочитанный файл лёг в память именно как набор функций.
2. Ингестия масштаба 100k токенов практична: 164 с на GPU, 391 тыс. embed-токенов, без вытеснений (capacity 8192).
3. Поиск на масштабе 1305 чанков находит СЕМАНТИЧЕСКИ СВЯЗАННЫЙ код, но не точные определения: на запрос про normalize_records/compute_total банк вернул методы классов, которые их вызывают (total(self)), и заголовок файла; на запрос про группировку — методы by_region. Exact-def hit = 0 во всех условиях.
4. Top-4 забивается почти одинаковыми чанками: три из четырёх notes на clean_mid — методы by_region разных классов. Нужна диверсификация выдачи (MMR/дедуп).
5. Преимущество памяти над no_memory на этих пробах не проявилось: хвост файла (40 строк) сам содержит методы с вызовами filter_active/compute_total/group_by, поэтому clean_total решается и без памяти. Единственная по-настоящему различающая проба — clean_mid (moving_average нет в хвосте): её не решил никто, банк вернул только group_by-usage, но не moving_average.
6. Валидность: почти весь код ast-валиден; два invalid — обрыв генерации на лимите 192 токена (артефакт метрики, не синтаксическая ошибка модели).

Вывод: механика (чанкинг → ингестия → связанный retrieval) масштабируется до 100k токенов, но для точного «вспомни нужную функцию» нужны: (а) диверсификация top-k, (б) гибридный скоринг (лексика по идентификаторам + семантика), (в) возможно ключ = сигнатура + docstring вместо всего тела. Это следующие итерации.

## Ограничения

- Одна сессия, greedy, n=1: проверка механики на масштабе, не статистика.
- Контекст значения — два предыдущих чанка файла (локальное соседство), не весь файл.
- Валидность — ast.parse сгенерированного блока, не исполнение кода.

Данные: `summary.json`, `<condition>.json` (полные ответы и injected notes).

# План: код-ветка памяти (codemem)

Отдельная система, НЕ переиспользует диалоговый `tmem`. Общая только идея
(центрирование + статическое хранение + динамический выбор). Архитектура —
из `для кода.md` + уточнения по рискам (обсуждение 15.11).

## Goal
Реализовать goal-conditioned память для написания кода: цель из запроса →
сжатие → дерево подцелей (центрирование, дендрограмма + проверка
самодостаточности, глубина ≤2) → перед каждым чанком план «чего хочу + что
нужно» → каскадный поиск (prefetch-предсказания + дерево, единый MMR-ранкинг)
→ генерация чанка → write-back по сходству с целью. Живой компонент =
предсказания (prefetch) + use-скоры зрелости; ключи статические, не мутируют.

## Решённые ограничения
- GPU: AMD RX 7900 XT, 20 GiB VRAM, torch 2.9.1+rocm6.3, transformers 5.22.0.
- Модель Qwen/Qwen3.5-4B (гибрид linear/full attention, hidden 2560, 32 слоя,
  full-attn 3/7/11/15/19/23/27/31), грузится AutoModelForImageTextToText,
  chat enable_thinking=False. Эмбеддинг = mid full-attn слой (default 15), mean-pool.
- HF_HUB_OFFLINE=1, только .venv/bin/python. scipy НЕТ → кластеризация своя.
- Код в отдельных папках: `codemem/`, `scripts_code/`, `tests_code/`, `results/code/`.
- Чистый encoder-decoder — отдельный staged-тест позже (t5-base smoke offline;
  CodeT5+ для вывода — скачка, по запросу). Не скрещивать с Qwen.

## Дизайн-решения (из обсуждения)
1. Цель очищать/сжимать; метрика верности cos(full, compressed). Альтернатива:
   эмбеддинг цели прямо из запроса без генерации текста.
2. Запрос = «написанное + желаемое». Цель маршрутизирует поддерево (грубо),
   потребность = центрированный residual-запрос внутри (мелко). Намерение
   живёт на правильном уровне и не вычитается центрированием.
3. Атомарность = дендрограмма + бинарный судья «оба подцеля самодостаточны?».
   Требовать зазор высоты слияния (cophenetic gap) прежде чем спрашивать судью.
4. Нерабочий код храним (инкрементальная кодовая база, метод класса не исполнить).
   Без execution-гейта на записи. Вместо: tiers зрелости WRITTEN→USED→VERIFIED
   (retroactive promotion) + use-скор по успешному применению. Ключи статические.
5. Prefetch — источник кандидатов, НЕ привилегированная секция: сливаем с
   tree-results в один MMR-ранжированный список (anchoring убран конструктивно).
6. Write-back: сходство с целью ≥ tau → ассимилировать в узел, иначе новая
   (решённая) цель/узел. Порог tau критичен (низкий→grab-bag, высокий→фрагментация).

## Фазы (каждая независимо проверяема)
- [x] Фаза 0: скелет пакета + config + vector (cosine/centroid/center/MMR) +
      cluster (дендрограмма) + unit-тесты (offline, CPU). Проверка: pytest зелёный.
- [x] Фаза 1 (spike, риск): model.py (load/embed/generate) + spike_embed.py —
      output_hidden_states на гибриде работает, [N,2560], слой 15, self-cos=1.
- [x] Фаза 2: tree (two-stage + write-back) + bank (static keys, tiers, use-score)
      + prefetch (TTL по шагу) + retrieve (единый каскад) + тесты на fake-эмбеддингах.
- [x] Фаза 3: goal (gen/compress/fidelity/decompose/judge) + planner (план-шаг +
      chunked generation + write-back). Smoke端到end на GPU: 2 чанка, fidelity 0.884.
- [x] Фаза 4: scripts_code/eval_retrieval.py на testbeds/taskq — 8 условий, 45
      запросов, bootstrap CI + paired. Итог: goal routing load-bearing (+9/-0),
      центрирование/диверсификация НЕ помогают на code (контраст с диалогом),
      prefetch +0, needroute коллапсирует. docs/CODE_BRANCH_FINDINGS.md.

## Риски / fallback
- output_hidden_states на гибриде может отличаться → spike до постройки всего;
  fallback: hook на блок или last hidden state.
- Качество сжатой цели / судьи самодостаточности шумное → мерить fidelity и
  согласие судьи с ораклом; depth cap 2 по умолчанию.
- Конфаунд масштаба для enc-dec → staged, within-space, честный framing.

## Verification
pytest tests_code/ (offline); spike_embed.py (GPU, коротко); eval_retrieval.py
(GPU, последовательная очередь, вотчер wait_and_report.sh). Коммиты по фазам
(Conventional Commits, English body), пуш.
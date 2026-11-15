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
- [ ] Фаза 0: скелет пакета + config + vector (cosine/centroid/center/MMR) +
      cluster (дендрограмма) + unit-тесты (offline, CPU). Проверка: pytest зелёный.
- [ ] Фаза 1 (spike, риск): model.py (load/embed/generate) + spike_embed.py —
      проверить output_hidden_states на гибриде, форму [N,2560], выбор слоя.
      Проверка: spike печатает формы, cos(a,a)=1, разные тексты <1. VRAM free first.
- [ ] Фаза 2: tree (two-stage + write-back) + bank (static keys, tiers, use-score)
      + prefetch (TTL по шагу) + retrieve (единый каскад) + тесты на fake-эмбеддингах.
      Проверка: pytest зелёный (ассимиляция/создание, MMR never-worse, TTL, tiers).
- [ ] Фаза 3: goal (gen/compress/fidelity/decompose/judge) + planner (план-шаг +
      chunked generation + write-back). Модель-зависимо. Проверка: smoke на 1 задаче.
- [ ] Фаза 4: scripts_code/eval_retrieval.py на testbeds/taskq — абляция
      flat-centered vs goal-routed vs +prefetch; метрики hits/reuse/def-hits.
      Проверка: results/code/<run>/ + отчёт. GPU, последовательно, вотчер.

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
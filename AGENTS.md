# Transformer — правила проекта

## Окружение
- Python — только `.venv/bin/python` (3.12, torch 2.9.1+rocm6.3, transformers 5.22.0). **scipy/sklearn НЕТ** — кластеризация своя (`codemem/cluster.py`), не добавляй зависимости молча.
- GPU: AMD RX 7900 XT, **20 GiB** VRAM, torch+ROCm, устройство называется `cuda`. (README и `docs/PLAN-fullscale-tests.md` пишут «XTX 24 GB» — устарело, сверено по sysfs.)
- Перед GPU-прогоном проверить свободную VRAM: `cat /sys/class/drm/card1/device/mem_info_vram_used` (`rocm-smi` не установлен; посторонние inference-серверы могут держать память). GPU не делить — прогоны строго последовательны.
- Модели уже в HF-кэше (`Qwen/Qwen3.5-4B`, gpt2, t5-base, Qwen2-1.5B); работать с `HF_HUB_OFFLINE=1`.

## Две ветки — не смешивать
- **Диалог**: `tmem/` + `tests/` + `scripts/`, результаты `results/<эксперимент>/` и `results/corpus/`.
- **Код**: `codemem/` + `tests_code/` + `scripts_code/` + `testbeds/`, результаты `results/code/`. codemem НЕ переиспользует tmem (общая только идея). Дизайн — `для кода.md` и `docs/PLAN-code-branch.md`, выводы — `docs/CODE_BRANCH_FINDINGS.md`.
- Канонический статус/выводы диалоговой ветки — `results/DIALOGUE_BRANCH.md`; общий журнал находок — `docs/TODO.md`. Перед новым экспериментом свериться с ними, чтобы не повторять уже проверенное.

## Термины — единый словарь
- **Живая память** (термин пользователя; синонимы: латентная память, латентный канал, банк активаций, K/V-инъекция) — ВСЕГДА наша собственная разработка: банк векторов `tmem/memory.py::Memory`, в который вливаются скрытые состояния средних full-attention слоёв (Qwen3.5-4B: 11/15/19/23, банк общий), и из которого `tmem/kvinject.py::KVMemoryAttention` инжектит векторы как дополнительные K/V на этапе генерации. Чистый вид в бенчмарках — стратегия `v4_kv_mid` (только инъекция, без текста). Всё, что пользователь говорит про «живую память», относится именно к этому. Сравнивалась с PyramidKV при разных настройках живой памяти: размер банка 128/512/1024 (прогон `shared-dev-v1`, `--strategy-set shared_dev`) и ширина инъекции kv_slots 8/16/32/64 + динамический гейт по косинусу (прогон `inj-v1`, `--strategy-set kv_sweep`) — итоги в `results/DIALOGUE_BRANCH.md`, не дублировать числа сюда.
- **Двухканальная память** — стратегия `v4_rag_kv`: живая память + текстовый RAG-банк.
- **RAG-банк / текстовый банк** — `tmem/semantic.py::RagBank` (тексты реплик + эмбеддинги, top-k в промпт). НЕ живая память.
- **KV-кэш** (classic / sliding / topk / snapkv / pyramidkv) — контрольные политики сжатия кэша, НЕ память.
- **Банк код-ветки** — `codemem/bank.py::CodeBank` (ретривал чанков кода в промпт, без K/V-инъекции) — отдельная система, не путать с живой памятью.

## Тесты
- `.venv/bin/python -m pytest -q` — офлайн, CPU, гоняет оба каталога (`testpaths = tests, tests_code`). Маркеры `integration` и `gpu` исключены по умолчанию (`pyproject.toml → addopts`).
- Одиночный: `.venv/bin/python -m pytest tests/test_memory.py -q` (или `::test_name`).
- `-m integration` — реальный GPT-2, нужна сеть; `-m gpu` — ТОЛЬКО по согласованию с пользователем.
- Unit-тесты используют крошечные детерминированные модели (`tests/conftest.py`), ничего не скачивают.

## Корпусный бенчмарк (`tmem.corpus_run`)
- Эталонная команда пилота (GPU, ROCm):
  ```bash
  HF_HUB_OFFLINE=1 .venv/bin/python -m tmem.corpus_run --preset pilot --model qwen35-4b \
      --dtype bfloat16 --batches 4 --run-id pilot-v1
  ```
- Пресеты: pilot=12, main=48, ablation=16, sweep=8 диалогов (план — `docs/PLAN-fullscale-tests.md`). `--strategy-set` (latent / shared / shared_dev / kv_sweep / full_core / full_self / full_rag / …) меняет набор стратегий; `--dry-run` — очередь, стратегии и ETA без GPU. Три полномасштабные ветки и их последовательный лаунчер описаны в `docs/PLAN-dialogue-live-fullscale.md`.
- Результаты: `results/corpus/<run_id>/` (JSON по стратегиям, `manifest.jsonl`, автоотчёт `INTERIM.md`). Повтор с тем же `--run-id` продолжает по манифесту (resume); `--batches N` — лимит батчей, `--only <id…>` — перегон конкретных.

## Статья генерируется — править генератор, не PAPER.md
- `results/report/PAPER.md`/PDF собираются из `scripts/paper_text.py` (+ `build_paper.py`, `paper_figures.py`, `build_report.py`). Правки формулировок — только в генераторах, затем пересборка.
- Сверка «статья ↔ код»: `.venv/bin/python scripts/verify_paper.py` (CPU, офлайн) и `tests/test_paper_verification.py`. Меняешь поведение алгоритма — синхронно правь генераторы: все опубликованные числа батареи зависят от него.

## Валидность старых данных
- Прогоны до коммита `68add5b` (15.11, фикс маски kvinject) имели инертный латентный канал в prefill: цифры v4 из battery100 (`results/qwen35_comparison/`) и `pilot-v1` условны (помечены ⚠️ pre-fix в `DIALOGUE_BRANCH.md`). Не цитировать как актуальные.

## Workflow: вотчеры на завершение (пользователь не мониторит)
- После запуска длинного прогона ВСЕГДА вешай вотчер, который сам собирает сводный отчёт и уведомляет пользователя. Пользователь не должен спрашивать «готово?».
- Механизм: прогон отвязанно (`setsid nohup … &`), поверх — фоновая shell-команда (`background=true`) `scripts/wait_and_report.sh "<pgrep-паттерн>" <отчёты…>`: ждёт окончания процессов и печатает сводку. Harness уведомляет о завершении → агент сам приносит результат.
- Очереди прогонов (`scripts/queue_*.sh`, `scripts_code/queue_*.sh`) ждут освобождения карты — не запускать два GPU-прогона параллельно.

## Коммиты
- Conventional Commits (`feat(memory): …`, `docs(corpus): …`), тело на английском, коммит по завершённой фазе, пуш в `origin/main` (работаем в `main` напрямую).

## Границы
- НЕ лазить в соседние проекты (например, `~/Documents/fine_tunes`) без прямой просьбы пользователя — ни читать, ни grep-ать, ни изменять. Тесты и данные этого проекта — здесь. См. глобальное правило в `~/.config/opencode/AGENTS.md`.

#!/usr/bin/env bash
# Вотчер на завершение: ждёт окончания процессов по pgrep-паттерну и печатает
# сводные отчёты. Запускать как фоновую shell-команду (background=true), чтобы
# harness уведомил агента о завершении и агент сам принёс результат пользователю.
#
# usage: wait_and_report.sh "<pgrep -f паттерн>" "<report1>" "<report2>" ...
# пример: wait_and_report.sh 'chunk_three_way|project_memory' \
#           results/chunk_code/project_taskq/REPORT.md
set -u
PATTERN="${1:?usage: wait_and_report.sh <pgrep-pattern> <reports...>}"
shift
echo "[watch $(date '+%F %T')] жду завершения: $PATTERN"
# pgrep -f матчит и cmdline самого вотчера (паттерн передан аргументом),
# поэтому исключаем процессы wait_and_report, иначе цикл ждёт сам себя вечно.
while pgrep -af "$PATTERN" | grep -v 'wait_and_report' | grep -q .; do sleep 30; done
echo "[watch $(date '+%F %T')] процессы завершены, сводка:"
for f in "$@"; do
  echo
  echo "===== $f ====="
  cat "$f" 2>/dev/null || echo "(нет файла: $f)"
done

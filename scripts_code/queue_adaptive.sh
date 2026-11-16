#!/usr/bin/env bash
# Очередь: a1 (confidence-adaptive routing) после освобождения GPU диалоговым
# corpus_run shared-v1 (PID 690228). GPU не делим, чужой прогон не убиваем.
# Ждём точечно по PID (kill -0), чтобы не матчить cmdline вотчера/очереди.
# Запуск detached: setsid nohup bash scripts_code/queue_adaptive.sh > results/code/adaptive_queue.log 2>&1 &
# Вотчер: bash scripts/wait_and_report.sh 'queue_adaptive|eval_adaptive' results/code/adaptive/a1/REPORT.md
set -u
cd /home/morgot/Documents/Transformer
export HF_HUB_OFFLINE=1
echo "[$(date '+%F %T')] жду завершения corpus_run PID 690228 (держит GPU)..."
while kill -0 690228 2>/dev/null; do sleep 30; done
echo "[$(date '+%F %T')] corpus_run завершён; запускаю a1 (confidence-adaptive)"
.venv/bin/python scripts_code/eval_adaptive.py --run-id a1
echo "[$(date '+%F %T')] a1 exit=$?"
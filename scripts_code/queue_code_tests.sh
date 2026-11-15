#!/usr/bin/env bash
# Очередь тестов код-ветки по плану (GPU не делим — строго последовательно).
# layer-sweep (11/19/23) + реализм routing (сгенерированная цель) + clone-flooding.
# Запуск detached: setsid nohup bash scripts_code/queue_code_tests.sh > results/code/queue.log 2>&1 &
# Вотчер: bash scripts/wait_and_report.sh 'queue_code_tests|eval_retrieval|clone_flood' <reports...>
set -u
cd /home/morgot/Documents/Transformer
export HF_HUB_OFFLINE=1
mkdir -p results/code
echo "[$(date '+%F %T')] queue start"
for L in 11 19 23; do
  echo "[$(date '+%F %T')] === layer $L ==="
  .venv/bin/python scripts_code/eval_retrieval.py --run-id "layer$L" --layer "$L"
  echo "[$(date '+%F %T')] layer $L exit=$?"
done
echo "[$(date '+%F %T')] === realism (generated-goal routing) ==="
.venv/bin/python scripts_code/eval_retrieval.py --run-id realism --layer 15 --realism
echo "[$(date '+%F %T')] realism exit=$?"
echo "[$(date '+%F %T')] === clone flooding ==="
.venv/bin/python scripts_code/clone_flood.py --run-id cf1
echo "[$(date '+%F %T')] clone_flood exit=$?"
echo "[$(date '+%F %T')] queue done"
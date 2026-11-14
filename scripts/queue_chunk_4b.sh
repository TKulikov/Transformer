#!/usr/bin/env bash
# Очередь (заказ 14.11): после текущего corpus_run — проверка чанковой памяти
# с контентным ключом на qwen35-4b (GPU, bf16), затем обновление отчёта.
# Запуск: setsid nohup bash scripts/queue_chunk_4b.sh <pid_текущего_прогона> \
#           > results/chunk_code_4b_nohup.log 2>&1 &
set -u
cd /home/morgot/Documents/Transformer
WAIT_PID="${1:?usage: queue_chunk_4b.sh <pid>}"
echo "[$(date '+%F %T')] жду завершения текущего прогона pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] текущий прогон завершён; запускаю проверку чанковой памяти на 4B"
HF_HUB_OFFLINE=1 .venv/bin/python scripts/chunk_code_eval_4b.py
rc=$?
echo "[$(date '+%F %T')] 4B eval exit=$rc"
.venv/bin/python - <<'EOF'
import sys
sys.path.insert(0, "scripts")
import night_watch as nw
nw.render_report(True, "проверка чанковой памяти на 4B завершена (очередь после текущего прогона)",
                 ["GPU: очередь пилота остановлена после текущего вызова — решение о продолжении за заказчиком.",
                  "4B: контентный ключ (активации блока) против запросного — см. секцию ниже."])
EOF
echo "[$(date '+%F %T')] отчёт обновлён: results/corpus/MORNING_REPORT.md"

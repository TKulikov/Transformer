#!/usr/bin/env bash
# Очередь: оценка памяти на проекте taskq после three-way и hybrid (GPU не делим).
set -u
cd /home/morgot/Documents/Transformer
echo "[$(date '+%F %T')] жду окончания three-way и hybrid..."
while pgrep -f 'chunk_three_way.py|chunk_hier_latent_eval.py' >/dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %T')] запускаю оценку на проекте taskq"
HF_HUB_OFFLINE=1 .venv/bin/python scripts/project_memory_eval.py
echo "[$(date '+%F %T')] project eval exit=$?"

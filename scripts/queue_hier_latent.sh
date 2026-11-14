#!/usr/bin/env bash
# Очередь: гибридная память (дерево+латент+TTL-gist) после трёхстороннего прогона.
# GPU не делим: ждём, пока chunk_three_way.py завершится.
set -u
cd /home/morgot/Documents/Transformer
echo "[$(date '+%F %T')] жду окончания chunk_three_way..."
while pgrep -f chunk_three_way.py >/dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %T')] three-way завершён; запускаю гибрид дерево+латент+TTL-gist"
HF_HUB_OFFLINE=1 .venv/bin/python scripts/chunk_hier_latent_eval.py
echo "[$(date '+%F %T')] hier-latent exit=$?"

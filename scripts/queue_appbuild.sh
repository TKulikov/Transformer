#!/usr/bin/env bash
# Очередь: app-building reuse-vs-write после goal-диалога. GPU не делим.
set -u
cd /home/morgot/Documents/Transformer
echo "[$(date '+%F %T')] жду окончания goal_dialogue_test..."
while pgrep -f goal_dialogue_test.py >/dev/null 2>&1; do sleep 30; done
echo "[$(date '+%F %T')] goal завершён; запускаю appbuild"
HF_HUB_OFFLINE=1 .venv/bin/python scripts/appbuild_test.py
echo "[$(date '+%F %T')] appbuild exit=$?"

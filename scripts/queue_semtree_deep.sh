#!/usr/bin/env bash
# Очередь: глубокое семантическое дерево после плоского semtree_eval.
# GPU не делим: ждём, пока semtree_eval.py завершится.
set -u
cd /home/morgot/Documents/Transformer
echo "[$(date '+%F %T')] жду окончания semtree_eval..."
while pgrep -f semtree_eval.py >/dev/null 2>&1; do sleep 30; done
echo "[$(date '+%F %T')] semtree_eval завершён; запускаю глубокое дерево"
HF_HUB_OFFLINE=1 .venv/bin/python scripts/semtree_deep_eval.py
echo "[$(date '+%F %T')] semtree_deep exit=$?"

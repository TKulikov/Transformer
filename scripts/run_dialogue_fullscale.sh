#!/usr/bin/env bash
# Последовательный запуск одной ветки полномасштабных диалоговых тестов.
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {core|self|rag} {gate|night|dry-run}" >&2
  exit 2
fi

TRACK=$1
MODE=$2

case "$TRACK" in
  core)
    STRATEGY_SET=full_core
    RUN_ID=full-core-v1
    NIGHT_BATCHES=4
    ;;
  self)
    STRATEGY_SET=full_self
    RUN_ID=full-self-v1
    NIGHT_BATCHES=3
    ;;
  rag)
    STRATEGY_SET=full_rag
    RUN_ID=full-rag-v1
    NIGHT_BATCHES=2
    ;;
  *)
    echo "unknown track: $TRACK (expected core, self or rag)" >&2
    exit 2
    ;;
esac

case "$MODE" in
  gate)
    BATCHES=1
    ;;
  night)
    BATCHES=$NIGHT_BATCHES
    ;;
  dry-run)
    BATCHES=${NIGHT_BATCHES}
    ;;
  *)
    echo "unknown mode: $MODE (expected gate, night or dry-run)" >&2
    exit 2
    ;;
esac

if [[ "$MODE" != dry-run ]]; then
  VRAM_FILE=/sys/class/drm/card1/device/mem_info_vram_used
  if [[ ! -r "$VRAM_FILE" ]]; then
    echo "cannot read VRAM usage: $VRAM_FILE" >&2
    exit 1
  fi
  read -r VRAM_USED < "$VRAM_FILE"
  MAX_VRAM_USED_BYTES=${MAX_VRAM_USED_BYTES:-1073741824}
  if ((VRAM_USED > MAX_VRAM_USED_BYTES)); then
    echo "GPU is busy: $((VRAM_USED / 1024 / 1024)) MiB VRAM already used; refusing to start" >&2
    exit 1
  fi
fi

ARGS=(
  -m tmem.corpus_run
  --preset main
  --strategy-set "$STRATEGY_SET"
  --model qwen35-4b
  --device cuda
  --dtype bfloat16
  --latency-repeats 5
  --batches "$BATCHES"
  --run-id "$RUN_ID"
)

if [[ "$MODE" == dry-run ]]; then
  ARGS+=(--dry-run)
fi

echo "[$(date '+%F %T')] track=$TRACK mode=$MODE run_id=$RUN_ID batches=$BATCHES"
HF_HUB_OFFLINE=1 .venv/bin/python "${ARGS[@]}"

if [[ "$MODE" != dry-run ]]; then
  echo "report: results/corpus/$RUN_ID/INTERIM.md"
fi

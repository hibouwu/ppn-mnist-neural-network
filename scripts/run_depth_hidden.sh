#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
OUT="$ROOT/output/depth_hidden"
mkdir -p "$OUT"

EPOCHS=20
BATCH=64
LR=0.01
ACT=relu
INIT=he

depths=("128" "128,128" "128,128,128")

for HS in "${depths[@]}"; do
  for SEED in 0 1 2; do
    echo "=== hidden_sizes=$HS seed=$SEED ==="

    rm -f "$BUILD/output/metrics.csv"

    (cd "$BUILD" && ./ppn_train \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH" \
      --learning_rate "$LR" \
      --hidden_sizes "$HS" \
      --seed "$SEED" \
      --activation "$ACT" \
      --init "$INIT")

    tag="${HS//,/x}"   # 128,128 -> 128x128
    mv -f "$BUILD/output/metrics.csv" "$OUT/metrics_${tag}_seed${SEED}.csv"
  done
done

echo "Done. Results saved in: $OUT"

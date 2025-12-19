#!/bin/bash

# Experiment: Learning Rate Impact
# Goal: Find the optimal learning rate & observe convergence/divergence behavior.
# Epochs: 20 (Long run to see stability)

BUILD_DIR="build"
EXE="$BUILD_DIR/ppn_train"
base_out="output"
EXP_DIR="$base_out/experiments/lr"

mkdir -p "$EXP_DIR"

# Hyperparameters to test
LRS=(0.025 0.01 0.0075 0.005 0.001)
SEEDS=(42 43 44)
EPOCHS=10

# Fixed parameters (Control Variates)
BATCH=64
HIDDEN=128
ACT="relu"
INIT="he"

echo "=================================================="
echo "Experiment: Learning Rate Study"
echo "Epochs: $EPOCHS | Batch: $BATCH | Hidden: $HIDDEN | Act: $ACT | Init: $INIT"
echo "Testing LRs: ${LRS[*]}"
echo "=================================================="

for lr in "${LRS[@]}"; do
    echo "------------------------------------------------"
    echo "Testing Learning Rate: $lr"
    echo "------------------------------------------------"
    
    for seed in "${SEEDS[@]}"; do
        NAME="lr_${lr}_seed${seed}"
        LOG_FILE="$EXP_DIR/${NAME}.log"
        CSV_FILE="$EXP_DIR/${NAME}.csv"
        
        echo "  > Run Seed $seed..."
        
        $EXE --epochs $EPOCHS \
             --learning_rate $lr \
             --batch_size $BATCH \
             --hidden_size $HIDDEN \
             --activation $ACT \
             --init $INIT \
             --seed $seed > "$LOG_FILE"
             
        if [ -f "$base_out/metrics.csv" ]; then
            mv "$base_out/metrics.csv" "$CSV_FILE"
            echo "    Saved: $CSV_FILE"
        else
            echo "    ERROR: metrics.csv missing for $NAME"
        fi
    done
done

echo "=================================================="
echo "Learning Rate Experiment Complete."
echo "Results saved to $EXP_DIR"

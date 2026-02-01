#!/bin/bash

# Configuration
EXE="./build/ppn_train"
EXP_DIR="output/ExperienceHPO/activation" # reusing activation dir for comparison
mkdir -p "$EXP_DIR"

# Hyperparameters to test
ACTIVATION="relu"
INIT="manual"
SEEDS=(42 43 44)
EPOCHS=20

# Fixed parameters
LR=0.01
BATCH=64
HIDDEN=128

echo "=================================================="
echo "Experiment: Initialization Study (Manual vs He)"
echo "Epochs: $EPOCHS | LR: $LR | Batch: $BATCH | Hidden: $HIDDEN"
echo "Testing: ${ACTIVATION} with ${INIT} init"
echo "=================================================="

for seed in "${SEEDS[@]}"; do
    echo "  > Run Seed $seed..."
    
    # Define output filename
    # Naming convention: act_relu_manual_seed42.csv
    # Previously we had act_relu_seed42 which implied he.
    # We will rename the new ones to act_relu_manual...
    
    OUT_NAME="act_relu_manual_seed${seed}"
    OUT_CSV="$EXP_DIR/${OUT_NAME}.csv"
    OUT_LOG="$EXP_DIR/${OUT_NAME}.log"
    
    # Run command
    $EXE --learning_rate $LR \
            --batch_size $BATCH \
            --hidden_size $HIDDEN \
            --activation $ACTIVATION \
            --init $INIT \
            --epochs $EPOCHS \
            --seed $seed \
            > "$OUT_LOG"
            
    # Move metrics file
    if [ -f "output/metrics.csv" ]; then
        mv "output/metrics.csv" "$OUT_CSV"
        echo "    Saved: $OUT_CSV"
    else
        echo "    ERROR: output/metrics.csv not found!"
    fi
done

echo "=================================================="
echo "Init Experiment Complete."

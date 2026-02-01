#!/bin/bash

# Configuration
EXE="./build/ppn_train"
EXP_DIR="output/ExperienceHPO/hidden"
mkdir -p "$EXP_DIR"

# Hyperparameters to test
HIDDEN_SIZES=(64 128 256)
SEEDS=(42 43 44)
EPOCHS=20

# Fixed parameters (Control Variates)
LR=0.01
BATCH=64        # Compromise choice from previous exp
ACTIVATION="relu"
INIT="he"

echo "=================================================="
echo "Experiment: Hidden Size Study"
echo "Epochs: $EPOCHS | LR: $LR | Batch: $BATCH | Act: $ACTIVATION | Init: $INIT"
echo "Testing Hidden Sizes: ${HIDDEN_SIZES[*]}"
echo "=================================================="

for hidden in "${HIDDEN_SIZES[@]}"; do
    echo "------------------------------------------------"
    echo "Testing Hidden Size: $hidden"
    echo "------------------------------------------------"
    
    for seed in "${SEEDS[@]}"; do
        echo "  > Run Seed $seed..."
        
        # Define output filename
        OUT_NAME="hidden_${hidden}_seed${seed}"
        OUT_CSV="$EXP_DIR/${OUT_NAME}.csv"
        OUT_LOG="$EXP_DIR/${OUT_NAME}.log"
        
        # Run command
        $EXE --learning_rate $LR \
             --batch_size $BATCH \
             --hidden_size $hidden \
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
done

echo "=================================================="
echo "Hidden Size Experiment Complete."
echo "Results saved to $EXP_DIR"

#!/bin/bash

# Configuration
EXE="./build/ppn_train"
EXP_DIR="output/ExperienceHPO/batch"
mkdir -p "$EXP_DIR"

# Hyperparameters to test
BATCH_SIZES=(16 32 64 128)
SEEDS=(42 43 44)
EPOCHS=20

# Fixed parameters (Control Variates)
LR=0.01         # Optimal from previous exp
HIDDEN=128
ACTIVATION="relu"
INIT="he"

echo "=================================================="
echo "Experiment: Batch Size Study"
echo "Epochs: $EPOCHS | LR: $LR | Hidden: $HIDDEN | Act: $ACTIVATION | Init: $INIT"
echo "Testing Batch Sizes: ${BATCH_SIZES[*]}"
echo "=================================================="

for batch in "${BATCH_SIZES[@]}"; do
    echo "------------------------------------------------"
    echo "Testing Batch Size: $batch"
    echo "------------------------------------------------"
    
    for seed in "${SEEDS[@]}"; do
        echo "  > Run Seed $seed..."
        
        # Define output filename
        OUT_NAME="batch_${batch}_seed${seed}"
        OUT_CSV="$EXP_DIR/${OUT_NAME}.csv"
        OUT_LOG="$EXP_DIR/${OUT_NAME}.log"
        
        # Run command
        $EXE --learning_rate $LR \
             --batch_size $batch \
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
done

echo "=================================================="
echo "Batch Size Experiment Complete."
echo "Results saved to $EXP_DIR"

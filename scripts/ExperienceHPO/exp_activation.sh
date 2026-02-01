#!/bin/bash

# Configuration
EXE="./build/ppn_train"
EXP_DIR="output/ExperienceHPO/activation"
mkdir -p "$EXP_DIR"

# Hyperparameters to test
ACTIVATIONS=("relu" "sigmoid" "tanh")
SEEDS=(42 43 44)
EPOCHS=20

# Fixed parameters (Control Variates)
LR=0.01
BATCH=64
HIDDEN=128

echo "=================================================="
echo "Experiment: Activation Function Study"
echo "Epochs: $EPOCHS | LR: $LR | Batch: $BATCH | Hidden: $HIDDEN"
echo "Testing Activations: ${ACTIVATIONS[*]}"
echo "=================================================="

for act in "${ACTIVATIONS[@]}"; do
    # Determine appropriate initialization
    if [ "$act" == "relu" ]; then
        INIT="he"
    else
        INIT="xavier"
    fi

    echo "------------------------------------------------"
    echo "Testing Activation: $act (Init: $INIT)"
    echo "------------------------------------------------"
    
    for seed in "${SEEDS[@]}"; do
        echo "  > Run Seed $seed..."
        
        # Define output filename
        OUT_NAME="act_${act}_seed${seed}"
        OUT_CSV="$EXP_DIR/${OUT_NAME}.csv"
        OUT_LOG="$EXP_DIR/${OUT_NAME}.log"
        
        # Run command
        $EXE --learning_rate $LR \
             --batch_size $BATCH \
             --hidden_size $HIDDEN \
             --activation $act \
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
echo "Activation Experiment Complete."
echo "Results saved to $EXP_DIR"

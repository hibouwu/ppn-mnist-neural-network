#!/bin/bash

# Ensure output directory exists
mkdir -p output

# Step 1: Recompile with PROFILE_MATMUL enabled
echo "=== Recompiling with PROFILE_MATMUL enabled ==="
cd build
cmake .. -DENABLE_PROFILE_MATMUL=ON -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..

# Define the implementations to test
impls=("blas" "ijk" "ikj" "blocked")
# impls=("blas" "blocked") # For faster testing if needed

echo "=== Starting Benchmark ===" > output/benchmark_results.txt
echo "Date: $(date)" >> output/benchmark_results.txt
echo "----------------------------------------" >> output/benchmark_results.txt

for impl in "${impls[@]}"; do
    echo "Running benchmark for: $impl"
    echo "----------------------------------------"
    
    # We use 'time' to measure the total execution time of the program (1 epoch)
    # We filter standard error to capture the PROFILE logs if needed, but for now we focus on total time.
    
    # Run the training for 1 epoch (ensure main.cpp is set to 1 epoch)
    # Using 'env' to set the variable for the process
    
    start_time=$(date +%s.%N)
    
    # Capture both stdout and stderr. 
    # Grep execution time for matmul if desired, or just measure total app time.
    # Here we measure total application time as it effectively shows the impact.
    output=$(MATMUL_IMPL=$impl ./build/ppn_train 2>&1)
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    # Extract training accuracy to ensure correctness (sanity check)
    acc=$(echo "$output" | grep "Final Test Accuracy" | awk '{print $NF}')
    
    echo "Implementation: $impl" >> output/benchmark_results.txt
    echo "Total Duration: $duration seconds" >> output/benchmark_results.txt
    echo "Final Accuracy: $acc" >> output/benchmark_results.txt
    echo "----------------------------------------" >> output/benchmark_results.txt
    
    echo "  -> Done in $duration seconds (Acc: $acc)"
done

echo "Benchmark completed. Results saved to output/benchmark_results.txt"
cat output/benchmark_results.txt

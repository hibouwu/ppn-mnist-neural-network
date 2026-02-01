#!/bin/bash

# Ensure output directory exists
mkdir -p output/ExperienceGEMM

# Recompile (DISABLE profiling logging to keep output clean)
echo "=== Recompiling for Rigorous Benchmark ==="
cd build
cmake .. -DENABLE_PROFILE_MATMUL=OFF -DCMAKE_BUILD_TYPE=Release
make -j4 test_benchmark_large
cd ..

# Define the implementations to test
configs=(
    "ijk:1"
    "ikj:1"
    "omp:4"
    "omp:8"
    "blas:8"
)

# STANDARD SQUARE SIZES (N x N)
sizes=(
    64
    128
    256
    512
)

# Target: 95% Confidence Interval Width < 5% of Mean
MAX_REPS=1000

echo "Implementation,Threads,Size,Reps,Mean,StdDev" > output/ExperienceGEMM/rigorous_comparison.csv
echo "=== Benchmarking Square Matrices (Adaptive: Max $MAX_REPS) ==="

for size in "${sizes[@]}"; do
    echo "--- Size: ${size}x${size} ---"

    for config in "${configs[@]}"; do
        IFS=':' read -r impl threads <<< "$config"
        
        echo "Running: $impl (Threads: $threads) | Size: ${size}x${size} | MaxReps: $MAX_REPS"
        
        # Export env vars
        export OMP_NUM_THREADS=$threads
        export OPENBLAS_NUM_THREADS=$threads
        
        # Run with max limit of 1000, but C++ will stop early if converged
        output=$(MATMUL_IMPL=$impl ./build/test_benchmark_large $size $size $size 1000)
        
        # Parse output format: "Done. Mean: X.XXXX s, StdDev: Y.YYYY s, Reps: N"
        mean=$(echo "$output" | grep "Done. Mean:" | awk '{print $3}')
        std=$(echo "$output" | grep "Done. Mean:" | awk '{print $6}')
        actual_reps=$(echo "$output" | grep "Done. Mean:" | awk '{print $9}')

        # Check if mean/std are valid
        if [ -z "$mean" ]; then mean="N/A"; fi
        if [ -z "$std" ]; then std="0.0"; fi
        if [ -z "$actual_reps" ]; then actual_reps="0"; fi
    
        echo "$impl,$threads,$size,$actual_reps,$mean,$std" >> output/ExperienceGEMM/rigorous_comparison.csv
    done
done

echo "Rigorous Benchmark completed. Results saved to output/ExperienceGEMM/rigorous_comparison.csv"
cat output/ExperienceGEMM/rigorous_comparison.csv

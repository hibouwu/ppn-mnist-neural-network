#!/bin/bash

# Ensure output directory exists
mkdir -p output

# Step 1: Recompile (DISABLE profiling logging to keep output clean, rely on C++ timer)
# We want to measure the full function call in test_benchmark_large.cpp
echo "=== Recompiling for Large Benchmark ==="
cd build
cmake .. -DENABLE_PROFILE_MATMUL=OFF -DCMAKE_BUILD_TYPE=Release
make -j4 test_benchmark_large
cd ..

# Define the implementations to test
# Define the implementations to test
impls=("blas" "ijk" "ikj" "blocked" "omp")
size=${1:-2048}

echo "=== Benchmarking Large Matrices (${size}x${size}) ===" > output/benchmark_large_results.txt
echo "Date: $(date)" >> output/benchmark_large_results.txt
echo "----------------------------------------" >> output/benchmark_large_results.txt

for impl in "${impls[@]}"; do

    # Determine iterations and OMP threads
    export OMP_NUM_THREADS=1
    if [ "$impl" == "ijk" ]; then
        ITERATIONS=3
    elif [ "$impl" == "omp" ]; then
        ITERATIONS=5
        export OMP_NUM_THREADS=16
    else
        ITERATIONS=5
    fi

    echo "Running benchmark for: $impl (Threads: $OMP_NUM_THREADS, Iterations: $ITERATIONS)"
    
    # Temp file for storing times
    > times.txt

    for ((i=1; i<=ITERATIONS; i++)); do
        # Run test_benchmark_large with size argument
        output=$(MATMUL_IMPL=$impl ./build/test_benchmark_large $size)
        
        # Extract time
        time_taken=$(echo "$output" | grep "Done in" | awk '{print $3}')
        echo "$time_taken" >> times.txt
        echo "  Run $i: $time_taken seconds"
    done
    
    # Compute mean and std dev
    result_line=$(python3 scripts/compute_stats.py < times.txt)
    
    echo "Implementation: $impl" >> output/benchmark_large_results.txt
    echo "Stats: $result_line seconds" >> output/benchmark_large_results.txt
    echo "----------------------------------------" >> output/benchmark_large_results.txt
    
    echo "  -> Result: $result_line"
    rm times.txt

done

echo "Large Benchmark completed. Results saved to output/benchmark_large_results.txt"
cat output/benchmark_large_results.txt

#!/bin/bash

# Ensure output directory exists
mkdir -p output

# Step 1: Recompile (DISABLE profiling logging to keep output clean, rely on C++ timer)
echo "=== Recompiling for OpenMP Benchmark ==="
cd build
cmake .. -DENABLE_PROFILE_MATMUL=OFF -DCMAKE_BUILD_TYPE=Release
make -j4 test_benchmark_large
cd ..

# Define thread counts to test
threads_list=(1 2 4 8 16)
size=${1:-2048}

echo "=== Benchmarking OpenMP Scaling (${size}x${size}) ===" > output/benchmark_omp_results.txt
echo "Date: $(date)" >> output/benchmark_omp_results.txt
echo "----------------------------------------" >> output/benchmark_omp_results.txt

impl="omp"
ITERATIONS=5

for t in "${threads_list[@]}"; do
    echo "Running benchmark for: $impl with OMP_NUM_THREADS=$t"
    
    # Temp file for storing times
    > times.txt

    for ((i=1; i<=ITERATIONS; i++)); do
        # Run test_benchmark_large with size argument
        output=$(OMP_NUM_THREADS=$t MATMUL_IMPL=$impl ./build/test_benchmark_large $size)
        
        # Extract time
        time_taken=$(echo "$output" | grep "Done in" | awk '{print $3}')
        echo "$time_taken" >> times.txt
        echo "  Run $i: $time_taken seconds"
    done
    
    # Compute mean and std dev
    result_line=$(python3 scripts/compute_stats.py < times.txt)
    
    echo "Threads: $t" >> output/benchmark_omp_results.txt
    echo "Stats: $result_line seconds" >> output/benchmark_omp_results.txt
    echo "----------------------------------------" >> output/benchmark_omp_results.txt
    
    echo "  -> Result: $result_line"
    rm times.txt

done

echo "OpenMP Benchmark completed. Results saved to output/benchmark_omp_results.txt"
cat output/benchmark_omp_results.txt

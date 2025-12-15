#!/bin/bash

# Ensure output directory exists
mkdir -p output

# Recompile just in case (ensure clean slate)
echo "=== Recompiling for BLAS Benchmark ==="
cd build
cmake .. -DENABLE_PROFILE_MATMUL=OFF -DCMAKE_BUILD_TYPE=Release
make -j4 test_benchmark_large
cd ..

threads_list=(1 2 4 8 16)
size=${1:-2048}

echo "=== Benchmarking BLAS Scaling (${size}x${size}) ===" > output/benchmark_blas_results.txt
echo "Date: $(date)" >> output/benchmark_blas_results.txt
echo "----------------------------------------" >> output/benchmark_blas_results.txt

impl="blas"
ITERATIONS=5

for t in "${threads_list[@]}"; do
    echo "Running benchmark for: $impl with Threads=$t"
    
    # Temp file for storing times
    > times.txt

    for ((i=1; i<=ITERATIONS; i++)); do
        # OpenBLAS respects these variables
        output=$(OPENBLAS_NUM_THREADS=$t OMP_NUM_THREADS=$t MATMUL_IMPL=$impl ./build/test_benchmark_large $size)
        
        # Extract time
        time_taken=$(echo "$output" | grep "Done in" | awk '{print $3}')
        echo "$time_taken" >> times.txt
        echo "  Run $i: $time_taken seconds"
    done
    
    # Compute mean and std dev
    result_line=$(python3 scripts/compute_stats.py < times.txt)
    
    echo "Threads: $t" >> output/benchmark_blas_results.txt
    echo "Stats: $result_line seconds" >> output/benchmark_blas_results.txt
    echo "----------------------------------------" >> output/benchmark_blas_results.txt
    
    echo "  -> Result: $result_line"
    rm times.txt

done

echo "BLAS Benchmark completed. Results saved to output/benchmark_blas_results.txt"
cat output/benchmark_blas_results.txt

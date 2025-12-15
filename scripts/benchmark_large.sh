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
# Configs to test: "Implementation:Threads"
# Note: ijk/ikj/blocked are single-threaded (Threads=1)
configs=(
    "ijk:1"
    "ikj:1"
    "blocked:1"
    "omp:4"
    "omp:8"
    "blas:8"
)

sizes=(64 128 256 512 1024 2048)
echo "Implementation,Threads,Size,Mean,StdDev" > output/impl_comparison.csv
echo "=== Benchmarking Implementations (All Sizes) ==="

for size in "${sizes[@]}"; do
    echo "--- Size: ${size}x${size} ---"
    
    # Determine repetitions based on size (copied from find_optimal_threads.sh)
    iterations=10
    warmups=3
    case $size in
        64)   iterations=100 ;;
        128)  iterations=50 ;;
        256)  iterations=25 ;;
    #    512)  iterations=20 ;;
    #    1024) iterations=12 ;;
    #    2048) iterations=6 ;;
        *)    iterations=5 ;;
    esac

    for config in "${configs[@]}"; do
        IFS=':' read -r impl threads <<< "$config"
        
        # Override iterations for really slow implementation (ijk) on large matrices to avoid timeout
        current_iters=$iterations
        if [ "$impl" == "ijk" ] && [ "$size" -ge 1024 ]; then
            current_iters=3
        fi
        
        echo "Running: $impl (Threads: $threads) | Size: $size | Reps: $current_iters"
        
        # Export env vars
        export OMP_NUM_THREADS=$threads
        export OPENBLAS_NUM_THREADS=$threads

        # Warm-up runs to avoid cold-start noise (results discarded)
        for ((w=1; w<=warmups; w++)); do
            MATMUL_IMPL=$impl ./build/test_benchmark_large $size >/dev/null
        done
        
        # Temp file for storing times
        timings=""
        
        for ((i=1; i<=current_iters; i++)); do
            output=$(MATMUL_IMPL=$impl ./build/test_benchmark_large $size)
            time_taken=$(echo "$output" | grep "Done in" | awk '{print $3}')
            timings="$timings$time_taken\n"
        done
        
        # Compute stats
        stats=$(echo -e "$timings" | python3 scripts/compute_stats.py)
        mean=$(echo "$stats" | awk '{print $1}')
        std=$(echo "$stats" | awk '{print $3}')
        
        # Fix N/A if stats failed
        if [ "$std" == "N/A" ] || [ -z "$std" ]; then std="0.00000000"; fi
    
        echo "$impl,$threads,$size,$mean,$std" >> output/impl_comparison.csv
    done
done

echo "Comparison Benchmark completed. Results saved to output/impl_comparison.csv"
cat output/impl_comparison.csv

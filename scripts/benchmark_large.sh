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
        512)  iterations=20 ;;
        1024) iterations=12 ;;
        2048) iterations=6 ;;
        *)    iterations=5 ;;
    esac

    for config in "${configs[@]}"; do
        IFS=':' read -r impl threads <<< "$config"
        
        # Initialize current_iters
        current_iters=$iterations

        if [ "$impl" == "ijk" ] && [ "$size" -ge 1024 ]; then
            current_iters=3
        fi
        
        echo "Running: $impl (Threads: $threads) | Size: $size | Reps: $current_iters"
        
        # Export env vars
        export OMP_NUM_THREADS=$threads
        export OPENBLAS_NUM_THREADS=$threads

        # Run ONCE with internal repetitions
        # The C++ binary now handles warmup, averaging, and stddev calculation
        output=$(MATMUL_IMPL=$impl ./build/test_benchmark_large $size $current_iters)
        
        # Parse output format: "Done. Mean: X.XXXX s, StdDev: Y.YYYY s"
        mean=$(echo "$output" | grep "Done. Mean:" | awk '{print $3}')
        std=$(echo "$output" | grep "Done. Mean:" | awk '{print $6}')

        # Check if mean/std are valid
        if [ -z "$mean" ]; then mean="N/A"; fi
        if [ -z "$std" ]; then std="0.0"; fi
    
        echo "$impl,$threads,$size,$mean,$std" >> output/impl_comparison.csv
    done
done

echo "Comparison Benchmark completed. Results saved to output/impl_comparison.csv"
cat output/impl_comparison.csv

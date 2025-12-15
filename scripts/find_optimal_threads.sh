#!/bin/bash

# Ensure output directory exists
mkdir -p output

# Recompile
cd build
cmake .. -DENABLE_PROFILE_MATMUL=OFF -DCMAKE_BUILD_TYPE=Release
make -j4 test_benchmark_large
cd ..

sizes=(64 128 256 512 1024 2048)
threads=(1 2 4 8 16)
implementations=("omp" "blas")

echo "Implementation,Size,Threads,Time,StdDev" > output/thread_scaling.csv
echo "=== Searching for Optimal Thread Counts (CSV Mode with Stats) ==="

for impl in "${implementations[@]}"; do
    for size in "${sizes[@]}"; do
        for t in "${threads[@]}"; do
            # Run specific config
            if [ "$impl" == "blas" ]; then
                export OPENBLAS_NUM_THREADS=$t
                export OMP_NUM_THREADS=$t # Fallback
            else
                export OMP_NUM_THREADS=$t
            fi
            
            # Determine number of iterations based on size to balance total time vs stability
            # Small matrices need more samples to reduce noise. Large matrices take longer so we run fewer.
            iterations=10
            case $size in
                64)   iterations=100 ;;
                128)  iterations=50 ;;
                256)  iterations=25 ;;
                512)  iterations=20 ;;
                1024) iterations=12 ;;
                2048) iterations=10 ;;
                *)    iterations=5 ;;
            esac
            
            # Run ONCE with internal repetitions
            # The C++ binary now handles warmup, averaging, and internal stddev calculation
            
            output=$(MATMUL_IMPL=$impl ./build/test_benchmark_large $size $iterations)
            
            # Parse output format: "Done. Mean: X.XXXX s, StdDev: Y.YYYY s"
            mean=$(echo "$output" | grep "Done. Mean:" | awk '{print $3}')
            std=$(echo "$output" | grep "Done. Mean:" | awk '{print $6}')

            # Check if mean/std are valid
            if [ -z "$mean" ]; then mean="N/A"; fi
            if [ -z "$std" ]; then std="0.0"; fi

            echo "$impl,$size,$t,$mean,$std" >> output/thread_scaling.csv
            echo "Recorded: $impl | Size: $size | Threads: $t | Reps: $iterations | Mean: $mean | Std: $std"
        done
    done
done

echo "Done. Saved using CSV format to output/thread_scaling.csv"
cat output/thread_scaling.csv

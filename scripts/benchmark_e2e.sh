#!/bin/bash

# Ensure output directory exists
mkdir -p output

# Step 1: Recompile in Release Mode (No Profiling)
# We want clean timing without gprof overhead
echo "=== Recompiling for E2E Benchmark (Release, No-GP) ==="
cd build
cmake .. -DENABLE_PROFILE_MATMUL=OFF -DCMAKE_BUILD_TYPE=Release
make -j8 ppn_train
cd ..

# Define implementations
implementations=("ijk" "ikj" "omp" "blas")

echo "Implementation,Mean,StdDev" > output/e2e_results.csv
echo "=== Benchmarking End-to-End Training (1 Epoch) ==="

for impl in "${implementations[@]}"; do
    # Adaptive repetitions
    reps=5
    if [ "$impl" == "ijk" ]; then
        reps=3 # Slow
    elif [ "$impl" == "omp" ] || [ "$impl" == "blas" ]; then
        reps=10 # Fast
    fi
    
    echo "Running: $impl | Reps: $reps"
    
    timings=""
    for (( i=1; i<=reps; i++ )); do
        # Use full threads for parallel versions
        export OMP_NUM_THREADS=8
        export OPENBLAS_NUM_THREADS=8
        
        # Measure Wall Time
        # /usr/bin/time -f "%e" prints seconds to stderr
        # We redirect stderr to stdout to capture it
        t_start=$(date +%s.%N)
        
        MATMUL_IMPL=$impl ./build/ppn_train > /dev/null
        
        t_end=$(date +%s.%N)
        duration=$(echo "$t_end - $t_start" | bc)
        
        echo "  Run $i: ${duration}s"
        timings="$timings$duration\n"
    done
    
    # Compute Stats
    stats=$(echo -e "$timings" | python3 scripts/compute_stats.py)
    # output format: "X.XXXXXX +/- Y.YYYYYY"
    mean=$(echo "$stats" | awk '{print $1}')
    std=$(echo "$stats" | awk '{print $3}')
    
    echo "$impl,$mean,$std" >> output/e2e_results.csv
done

echo "E2E Benchmark Completed."
cat output/e2e_results.csv

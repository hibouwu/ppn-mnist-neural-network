#!/bin/bash

# Ensure output directory exists
mkdir -p output

# Recompile to ensure release mode
echo "=== Recompiling for Affinity Benchmark ==="
cd build
cmake .. -DENABLE_PROFILE_MATMUL=OFF -DCMAKE_BUILD_TYPE=Release
make -j8 test_benchmark_large
cd ..

# Define parameters
size=28
iterations=100000 # High repetitions for tiny matrix

implementations=("omp" "blas")
thread_counts=(1 2 4 6 8 16)
affinity_modes=("default" "close" "spread")

echo "Implementation,Threads,Affinity,Mean,StdDev" > output/affinity_comparison.csv
echo "=== Benchmarking Affinity for Size ${size}x${size} ==="

for impl in "${implementations[@]}"; do
    for t in "${thread_counts[@]}"; do
        for aff in "${affinity_modes[@]}"; do
            
            # Reset Env Vars
            unset OMP_PLACES
            unset OMP_PROC_BIND
            unset OPENBLAS_MAIN_FREE
            
            # Set Thread Count
            export OMP_NUM_THREADS=$t
            export OPENBLAS_NUM_THREADS=$t
            
            # Set Affinity
            if [ "$aff" == "close" ]; then
                export OMP_PLACES=cores
                export OMP_PROC_BIND=close
            elif [ "$aff" == "spread" ]; then
                export OMP_PLACES=cores
                export OMP_PROC_BIND=spread
            fi
            
            # For BLAS, we might need to tell it to not manage threads if we want OMP vars to apply
            # But standard OpenBLAS often ignores OMP_PROC_BIND unless built with OpenMP.
            # We test it anyway as requested.
            
            # Run Benchmark
            echo "Running: $impl | Threads: $t | Affinity: $aff"
            
            output=$(MATMUL_IMPL=$impl ./build/test_benchmark_large $size $iterations)
            
            # Parse output
            mean=$(echo "$output" | grep "Done. Mean:" | awk '{print $3}')
            std=$(echo "$output" | grep "Done. Mean:" | awk '{print $6}')

            if [ -z "$mean" ]; then mean="N/A"; fi
            if [ -z "$std" ]; then std="0.0"; fi
            
            echo "$impl,$t,$aff,$mean,$std" >> output/affinity_comparison.csv
        done
    done
done

echo "Affinity Benchmark Completed. Results in output/affinity_comparison.csv"
cat output/affinity_comparison.csv

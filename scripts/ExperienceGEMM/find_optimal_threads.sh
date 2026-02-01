#!/bin/bash

# Thread Scaling Benchmark
# Pure Thread Scaling (No internal affinity logic)
# Run this script via taskset if pinning is desired.

# Result CSV
OUTPUT_CSV="output/ExperienceGEMM/thread_scaling.csv"

# ==========================================
# 0. Rigorous Environment Setup
# ==========================================

# Default frequencies
FREQ_TARGET="4000MHz"

setup_env() {
    echo "=== [Setup] Configuring Execution Environment ==="
    if command -v cpupower &> /dev/null; then
        echo "  -> Setting CPU Governor to 'performance'..."
        sudo cpupower frequency-set -g performance > /dev/null
        echo "  -> Locking CPU Frequency to $FREQ_TARGET..."
        sudo cpupower frequency-set -u $FREQ_TARGET -d $FREQ_TARGET > /dev/null
        current_freq=$(cpupower frequency-info | grep "current CPU frequency" | head -n 1)
        echo "  -> $current_freq"
    else
        echo "  [WARNING] 'cpupower' not found."
    fi
}

restore_env() {
    echo ""
    echo "=== [Teardown] Restoring Environment ==="
    if command -v cpupower &> /dev/null; then
        sudo cpupower frequency-set -g powersave > /dev/null
    fi
    echo "Done."
}

trap restore_env EXIT
setup_env

# ==========================================
# 1. Compilation
# ==========================================
echo "=== Compiling Benchmark Tool ==="
cd build
cmake .. -DENABLE_PROFILE_MATMUL=OFF -DCMAKE_BUILD_TYPE=Release
make -j4 test_benchmark_large
cd ..

# 2. Configuration
SIZES=(64 128 256 512)
THREADS_LIST=(1 2 4 8 16)
IMPLS=("omp" "blas")
REPS=2000  # Augmenté à 5000 pour lisser la variabilité (voir Exp 6)

HAS_PERF=false
if command -v perf &> /dev/null; then
    HAS_PERF=true
fi

# Metric: Process-level (1500 reps x 3 runs)
echo "Implementation,Size,Threads,Time_us,StdDev_us,Instructions,Instr_StdDev,Cycles,Cycles_StdDev,IPC,IPC_StdDev,CS,CS_StdDev,CacheMisses,Cache_StdDev,CpuMigrations,Mig_StdDev,Reps" > $OUTPUT_CSV
echo "=== Searching for Optimal Thread Counts ==="
echo "Results will be saved to: $OUTPUT_CSV"

run_test() {
    local impl=$1
    local size=$2
    local t=$3

    export OMP_NUM_THREADS=$t
    export OPENBLAS_NUM_THREADS=$t
    export MATMUL_IMPL=$impl
    
    # Force affinity for stability
    export OMP_PROC_BIND=true
    export OMP_PLACES=cores

    echo "---------------------------------------------------"
    echo "Running: $impl | Size: $size | Threads: $t"

    local bench_cmd="./build/test_benchmark_large $size $size $size $REPS"
    
    # Run command (No internal taskset)
    local output=$($bench_cmd)
    
    local mean=$(echo "$output" | grep "Mean:" | awk '{print $3}')
    local std=$(echo "$output" | grep "Mean:" | awk '{print $6}')
    local actual_reps=$(echo "$output" | grep "Mean:" | awk '{print $9}')

    mean_us=$(awk -v val="$mean" 'BEGIN {printf "%.2f", val * 1000000}')
    std_us=$(awk -v val="$std" 'BEGIN {printf "%.2f", val * 1000000}')

    if [ -z "$mean" ]; then mean_us="ERR"; std_us="ERR"; fi
    if [ -z "$actual_reps" ]; then actual_reps="0"; fi

    local perf_reps=$REPS

    local instructions="N/A"
    local instr_std="N/A"
    local cycles="N/A"
    local cycles_std="N/A"
    local ipc="N/A"
    local ipc_std="N/A"
    local ctx="N/A"
    local ctx_std="N/A"
    local cache_miss="N/A"
    local cache_std="N/A"
    local cpu_mig="N/A"
    local mig_std="N/A"

    if [ "$HAS_PERF" = true ]; then
        # Single perf run: Process-level metrics
        local perf_output=$( perf stat -x, -r 3 -e instructions,cycles,context-switches,cpu-migrations,cache-misses,cache-references \
            ./build/test_benchmark_large $size $size $size $perf_reps 2>&1 )
        
        # CSV format: value,unit,event,variance%,runtime,pct
        
        # Extract Instructions
        local instr_line=$(echo "$perf_output" | grep "instructions")
        instructions=$(echo "$instr_line" | awk -F, '{print $1}')
        local instr_variance_pct=$(echo "$instr_line" | awk -F, '{print $4}' | sed 's/%//')
        if [ -n "$instructions" ] && [ -n "$instr_variance_pct" ]; then
            instr_std=$(awk -v mean="$instructions" -v var="$instr_variance_pct" 'BEGIN {printf "%.2f", mean * var / 100}')
        fi
        
        # Extract Cycles
        local cycles_line=$(echo "$perf_output" | grep "cycles")
        cycles=$(echo "$cycles_line" | awk -F, '{print $1}')
        local cycles_variance_pct=$(echo "$cycles_line" | awk -F, '{print $4}' | sed 's/%//')
        if [ -n "$cycles" ] && [ -n "$cycles_variance_pct" ]; then
            cycles_std=$(awk -v mean="$cycles" -v var="$cycles_variance_pct" 'BEGIN {printf "%.2f", mean * var / 100}')
        fi
        
        # Calculate true IPC = instructions / cycles
        if [ -n "$instructions" ] && [ -n "$cycles" ] && [ "$cycles" != "0" ]; then
            ipc=$(awk -v instr="$instructions" -v cyc="$cycles" 'BEGIN {printf "%.4f", instr / cyc}')
            # IPC StdDev using error propagation
            if [ -n "$instr_std" ] && [ -n "$cycles_std" ]; then
                ipc_std=$(awk -v ipc_val="$ipc" -v instr="$instructions" -v instr_s="$instr_std" \
                              -v cyc="$cycles" -v cyc_s="$cycles_std" \
                              'BEGIN {
                                  rel_i = instr_s / instr
                                  rel_c = cyc_s / cyc
                                  rel_ipc = sqrt(rel_i*rel_i + rel_c*rel_c)
                                  printf "%.4f", ipc_val * rel_ipc
                              }')
            fi
        fi
        
        # Extract context-switches
        local ctx_line=$(echo "$perf_output" | grep "context-switches")
        ctx=$(echo "$ctx_line" | awk -F, '{print $1}')
        local ctx_variance_pct=$(echo "$ctx_line" | awk -F, '{print $4}' | sed 's/%//')
        if [ -n "$ctx" ] && [ -n "$ctx_variance_pct" ]; then
            ctx_std=$(awk -v mean="$ctx" -v var="$ctx_variance_pct" 'BEGIN {printf "%.2f", mean * var / 100}')
        fi
        
        # Extract cache-misses
        local cache_line=$(echo "$perf_output" | grep "cache-misses")
        cache_miss=$(echo "$cache_line" | awk -F, '{print $1}')
        local cache_variance_pct=$(echo "$cache_line" | awk -F, '{print $4}' | sed 's/%//')
        if [ -n "$cache_miss" ] && [ -n "$cache_variance_pct" ]; then
            cache_std=$(awk -v mean="$cache_miss" -v var="$cache_variance_pct" 'BEGIN {printf "%.2f", mean * var / 100}')
        fi
        
        # Extract cpu-migrations
        local mig_line=$(echo "$perf_output" | grep "cpu-migrations")
        cpu_mig=$(echo "$mig_line" | awk -F, '{print $1}')
        local mig_variance_pct=$(echo "$mig_line" | awk -F, '{print $4}' | sed 's/%//')
        if [ -n "$cpu_mig" ] && [ -n "$mig_variance_pct" ]; then
            mig_std=$(awk -v mean="$cpu_mig" -v var="$mig_variance_pct" 'BEGIN {printf "%.2f", mean * var / 100}')
        fi
    fi

    echo "  -> Time: ${mean_us} us | Instr: ${instructions}±${instr_std} | Cycles: ${cycles}±${cycles_std} | IPC: ${ipc}±${ipc_std} | CS: ${ctx}±${ctx_std} | CpuMig: ${cpu_mig}±${mig_std} | CacheMiss: ${cache_miss}±${cache_std} | Reps: ${actual_reps} (perf: ${perf_reps}x3)"
    
    # Save to CSV
    echo "$impl,$size,$t,$mean_us,$std_us,$instructions,$instr_std,$cycles,$cycles_std,$ipc,$ipc_std,$ctx,$ctx_std,$cache_miss,$cache_std,$cpu_mig,$mig_std,$actual_reps" >> $OUTPUT_CSV
}

for impl in "${IMPLS[@]}"; do
    for size in "${SIZES[@]}"; do
        for t in "${THREADS_LIST[@]}"; do
            run_test "$impl" "$size" "$t"
        done
    done
done

echo "Done. Results saved to $OUTPUT_CSV"

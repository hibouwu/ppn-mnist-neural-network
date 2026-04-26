#!/bin/bash

# AVX-512 thread scaling benchmark.
# Mirrors scripts/ExperienceGEMM/find_optimal_threads.sh but focuses on the
# custom AVX-512 GotoBLAS path only.
#
# Optional overrides:
#   AVX512_KERNEL=avx512_8x32
#   AVX512_MC=8
#   AVX512_NC=448
#   AVX512_KC=160
#   SIZES="64 128 256 512"
#   THREADS_LIST="1 2 4 8 16"
#   REPS=2000
#
# Run this script via taskset if pinning/isolation is desired.

set -u

OUTPUT_CSV="${OUTPUT_CSV:-output/ExperienceGEMM/thread_scaling_avx512.csv}"
FREQ_TARGET="${FREQ_TARGET:-4000MHz}"
AVX512_KERNEL="${AVX512_KERNEL:-avx512_8x32}"
AVX512_MC="${AVX512_MC:-8}"
AVX512_NC="${AVX512_NC:-448}"
AVX512_KC="${AVX512_KC:-160}"
REPS="${REPS:-2000}"

read -r -a SIZES_ARR <<< "${SIZES:-64 128 256 512}"
read -r -a THREADS_ARR <<< "${THREADS_LIST:-1 2 4 8 16}"

setup_env() {
    echo "=== [Setup] Configuring Execution Environment ==="
    if command -v cpupower &> /dev/null; then
        echo "  -> Setting CPU Governor to 'performance'..."
        sudo cpupower frequency-set -g performance > /dev/null
        echo "  -> Locking CPU Frequency to $FREQ_TARGET..."
        sudo cpupower frequency-set -u "$FREQ_TARGET" -d "$FREQ_TARGET" > /dev/null
        local current_freq
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

if ! lscpu | grep -q "avx512f"; then
    echo "[ERROR] This machine does not report AVX-512F support."
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"
setup_env

echo "=== Compiling Benchmark Tool ==="
cd build
cmake .. -DENABLE_PROFILE_MATMUL=OFF -DCMAKE_BUILD_TYPE=Release
make -j4 test_benchmark_large
cd ..

HAS_PERF=false
if command -v perf &> /dev/null; then
    HAS_PERF=true
fi

echo "Implementation,KernelShape,MC,NC,KC,Size,Threads,Time_us,StdDev_us,Instructions,Instr_StdDev,Cycles,Cycles_StdDev,IPC,IPC_StdDev,CS,CS_StdDev,CacheMisses,Cache_StdDev,CpuMigrations,Mig_StdDev,Reps" > "$OUTPUT_CSV"

echo "=== AVX-512 Thread Sweep ==="
echo "Implementation : omp_gotoblas_avx512"
echo "Kernel shape   : $AVX512_KERNEL"
echo "MC/NC/KC       : $AVX512_MC / $AVX512_NC / $AVX512_KC"
echo "Sizes          : ${SIZES_ARR[*]}"
echo "Threads        : ${THREADS_ARR[*]}"
echo "Repetitions    : $REPS"
echo "Output CSV     : $OUTPUT_CSV"

run_test() {
    local size=$1
    local threads=$2

    export MATMUL_IMPL=omp_gotoblas_avx512
    export MATMUL_GOTO_KERNEL="$AVX512_KERNEL"
    export MATMUL_MC="$AVX512_MC"
    export MATMUL_NC="$AVX512_NC"
    export MATMUL_KC="$AVX512_KC"
    export OMP_NUM_THREADS="$threads"
    export OPENBLAS_NUM_THREADS=1
    export OMP_PROC_BIND=true
    export OMP_PLACES=cores

    echo "---------------------------------------------------"
    echo "Running: omp_gotoblas_avx512 | Kernel: $AVX512_KERNEL | MC/NC/KC: $AVX512_MC/$AVX512_NC/$AVX512_KC | Size: $size | Threads: $threads"

    local bench_cmd="./build/test_benchmark_large $size $size $size $REPS"
    local output
    output=$($bench_cmd)

    local mean
    local std
    local actual_reps
    mean=$(echo "$output" | grep "Mean:" | awk '{print $3}')
    std=$(echo "$output" | grep "Mean:" | awk '{print $6}')
    actual_reps=$(echo "$output" | grep "Mean:" | awk '{print $9}')

    local mean_us
    local std_us
    mean_us=$(awk -v val="$mean" 'BEGIN {printf "%.2f", val * 1000000}')
    std_us=$(awk -v val="$std" 'BEGIN {printf "%.2f", val * 1000000}')

    if [ -z "$mean" ]; then mean_us="ERR"; std_us="ERR"; fi
    if [ -z "$actual_reps" ]; then actual_reps="0"; fi

    local perf_reps="$REPS"
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
        local perf_output
        perf_output=$(perf stat -x, -r 3 -e instructions,cycles,context-switches,cpu-migrations,cache-misses,cache-references \
            ./build/test_benchmark_large "$size" "$size" "$size" "$perf_reps" 2>&1)

        local instr_line
        instr_line=$(echo "$perf_output" | grep "instructions")
        instructions=$(echo "$instr_line" | awk -F, '{print $1}')
        local instr_variance_pct
        instr_variance_pct=$(echo "$instr_line" | awk -F, '{print $4}' | sed 's/%//')
        if [ -n "$instructions" ] && [ -n "$instr_variance_pct" ]; then
            instr_std=$(awk -v mean="$instructions" -v var="$instr_variance_pct" 'BEGIN {printf "%.2f", mean * var / 100}')
        fi

        local cycles_line
        cycles_line=$(echo "$perf_output" | grep "cycles")
        cycles=$(echo "$cycles_line" | awk -F, '{print $1}')
        local cycles_variance_pct
        cycles_variance_pct=$(echo "$cycles_line" | awk -F, '{print $4}' | sed 's/%//')
        if [ -n "$cycles" ] && [ -n "$cycles_variance_pct" ]; then
            cycles_std=$(awk -v mean="$cycles" -v var="$cycles_variance_pct" 'BEGIN {printf "%.2f", mean * var / 100}')
        fi

        if [ -n "$instructions" ] && [ -n "$cycles" ] && [ "$cycles" != "0" ]; then
            ipc=$(awk -v instr="$instructions" -v cyc="$cycles" 'BEGIN {printf "%.4f", instr / cyc}')
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

        local ctx_line
        ctx_line=$(echo "$perf_output" | grep "context-switches")
        ctx=$(echo "$ctx_line" | awk -F, '{print $1}')
        local ctx_variance_pct
        ctx_variance_pct=$(echo "$ctx_line" | awk -F, '{print $4}' | sed 's/%//')
        if [ -n "$ctx" ] && [ -n "$ctx_variance_pct" ]; then
            ctx_std=$(awk -v mean="$ctx" -v var="$ctx_variance_pct" 'BEGIN {printf "%.2f", mean * var / 100}')
        fi

        local cache_line
        cache_line=$(echo "$perf_output" | grep "cache-misses")
        cache_miss=$(echo "$cache_line" | awk -F, '{print $1}')
        local cache_variance_pct
        cache_variance_pct=$(echo "$cache_line" | awk -F, '{print $4}' | sed 's/%//')
        if [ -n "$cache_miss" ] && [ -n "$cache_variance_pct" ]; then
            cache_std=$(awk -v mean="$cache_miss" -v var="$cache_variance_pct" 'BEGIN {printf "%.2f", mean * var / 100}')
        fi

        local mig_line
        mig_line=$(echo "$perf_output" | grep "cpu-migrations")
        cpu_mig=$(echo "$mig_line" | awk -F, '{print $1}')
        local mig_variance_pct
        mig_variance_pct=$(echo "$mig_line" | awk -F, '{print $4}' | sed 's/%//')
        if [ -n "$cpu_mig" ] && [ -n "$mig_variance_pct" ]; then
            mig_std=$(awk -v mean="$cpu_mig" -v var="$mig_variance_pct" 'BEGIN {printf "%.2f", mean * var / 100}')
        fi
    fi

    echo "  -> Time: ${mean_us} us | Instr: ${instructions}±${instr_std} | Cycles: ${cycles}±${cycles_std} | IPC: ${ipc}±${ipc_std} | CS: ${ctx}±${ctx_std} | CpuMig: ${cpu_mig}±${mig_std} | CacheMiss: ${cache_miss}±${cache_std} | Reps: ${actual_reps} (perf: ${perf_reps}x3)"

    echo "omp_gotoblas_avx512,$AVX512_KERNEL,$AVX512_MC,$AVX512_NC,$AVX512_KC,$size,$threads,$mean_us,$std_us,$instructions,$instr_std,$cycles,$cycles_std,$ipc,$ipc_std,$ctx,$ctx_std,$cache_miss,$cache_std,$cpu_mig,$mig_std,$actual_reps" >> "$OUTPUT_CSV"
}

for size in "${SIZES_ARR[@]}"; do
    for threads in "${THREADS_ARR[@]}"; do
        run_test "$size" "$threads"
    done
done

echo "Done. Results saved to $OUTPUT_CSV"

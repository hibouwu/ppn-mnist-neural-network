#!/bin/bash

# Build directory check
if [ ! -d "build" ]; then
    echo "Error: build directory not found. Please run cmake and make first."
    exit 1
fi

# Check if we are in scripts dir or root
if [ -d "../build" ]; then
    cd ../build || exit 1
elif [ -d "build" ]; then
    cd build || exit 1
else
    echo "Error: Could not find build directory"
    exit 1
fi

echo "Starting benchmark for implementations: blas, ijk, ikj, blocked, omp..."
echo "This may take a few minutes."
echo ""

# Table Header
printf "%-10s | %-15s | %-15s | %-30s\n" "Impl" "Wall-Time" "CPU-Time" "Top-Gprof-Function"
printf "%s\n" "--------------------------------------------------------------------------------------"

# Loop through each implementation
for impl in blas ijk ikj blocked omp; do
    export MATMUL_IMPL=$impl
    
    # 1. Run training (redirect stdout to log, stderr to time log)
    # Using specific constraints to keep runtimes reasonable
    /usr/bin/time -v ./ppn_train --epochs 1 --batch_size 64 --hidden_size 128 --data_dir ../mnist > "train_${impl}.log" 2> "time_${impl}.log"
    
    # 2. Generate gprof report
    gprof ./ppn_train gmon.out > "gprof_${impl}.txt"
    
    # 3. Extract Metrics
    # Wall Clock Time (format h:mm:ss or m:ss)
    wc_time=$(grep "Elapsed (wall clock)" "time_${impl}.log" | awk '{print $8}')
    
    # CPU Time (seconds)
    cpu_time=$(grep "User time" "time_${impl}.log" | awk '{print $4}')
    
    # Top function from Gprof (first data line after header)
    top_func=$(awk '
        /^ +%/ { header_found=1; next }
        header_found && $1 ~ /^[0-9]/ { print $7; exit }
    ' "gprof_${impl}.txt")
    
    # 4. Print Row
    printf "%-10s | %-15s | %-15s | %-30s\n" "$impl" "$wc_time" "$cpu_time" "$top_func"
done

echo ""
echo " detailed logs: build/train_*.log"
echo " profile reports: build/gprof_*.txt"

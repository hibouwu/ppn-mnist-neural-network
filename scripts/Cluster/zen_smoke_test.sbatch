#!/usr/bin/env bash
#SBATCH --job-name=ppn-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=ppn-smoke-%j.out
#SBATCH --error=ppn-smoke-%j.err

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_ID="${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}"
LOG_ROOT="${PPN_CLUSTER_LOG_DIR:-${PROJECT_ROOT}/output/cluster_smoke/${JOB_ID}}"
BUILD_DIR="${PPN_BUILD_DIR:-${PROJECT_ROOT}/build-zen-smoke}"
PROFILE="${PPN_SMOKE_PROFILE:-quick}"
RUN_DATA_TESTS="${PPN_RUN_DATA_TESTS:-auto}"
RUN_MPI="${PPN_RUN_MPI:-0}"
ENABLE_ONEDNN="${PPN_ENABLE_ONEDNN_CONV_BACKEND:-OFF}"
RUN_ONEDNN_PARITY="${PPN_RUN_ONEDNN_PARITY:-auto}"
ENABLE_GPU="${PPN_ENABLE_GPU_SMOKE_TESTS:-OFF}"
RUN_GPU="${PPN_RUN_GPU_SMOKE:-0}"
RUN_CHECKPOINT="${PPN_RUN_CHECKPOINT_SMOKE:-auto}"
GENERATOR="${PPN_CMAKE_GENERATOR:-}"

mkdir -p "${LOG_ROOT}"
exec > >(tee "${LOG_ROOT}/run.log") 2>&1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

log() {
    printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

run() {
    log "$*"
    "$@"
}

expect_file() {
    local path="$1"
    if [[ ! -s "${path}" ]]; then
        echo "Expected non-empty file was not produced: ${path}" >&2
        return 1
    fi
}

have_mnist() {
    [[ -f "${PROJECT_ROOT}/mnist/train-images-idx3-ubyte" ]] &&
    [[ -f "${PROJECT_ROOT}/mnist/train-labels-idx1-ubyte" ]] &&
    [[ -f "${PROJECT_ROOT}/mnist/t10k-images-idx3-ubyte" ]] &&
    [[ -f "${PROJECT_ROOT}/mnist/t10k-labels-idx1-ubyte" ]]
}

log "PPN Zen smoke test"
echo "Project root: ${PROJECT_ROOT}"
echo "Build dir:    ${BUILD_DIR}"
echo "Log dir:      ${LOG_ROOT}"
echo "Profile:      ${PROFILE}"
echo "Run data:     ${RUN_DATA_TESTS}"
echo "Run MPI:      ${RUN_MPI}"
echo "oneDNN:       ${ENABLE_ONEDNN}"
echo "oneDNN probe: ${RUN_ONEDNN_PARITY}"
echo "GPU targets:  ${ENABLE_GPU}"
echo "Run GPU:      ${RUN_GPU}"
echo "Checkpoint:   ${RUN_CHECKPOINT}"
echo "Host:         $(hostname)"
echo "User:         ${USER:-unknown}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Slurm job id: ${SLURM_JOB_ID}"
    echo "Slurm nodes:  ${SLURM_JOB_NUM_NODES:-unknown}"
    echo "Slurm tasks:  ${SLURM_NTASKS:-unknown}"
    echo "Slurm cpus/task: ${SLURM_CPUS_PER_TASK:-unknown}"
    if command -v srun >/dev/null 2>&1; then
        run srun --ntasks=1 --cpus-per-task=1 hostname
    fi
fi

if [[ -n "${PPN_MODULES:-}" ]]; then
    log "Loading modules from PPN_MODULES=${PPN_MODULES}"
    # shellcheck disable=SC2206
    MODULES_TO_LOAD=(${PPN_MODULES})
    for module_name in "${MODULES_TO_LOAD[@]}"; do
        run module load "${module_name}"
    done
fi

run cmake --version
run "${CXX:-c++}" --version

CMAKE_ARGS=(
    -S "${PROJECT_ROOT}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DENABLE_MPI=OFF
    -DENABLE_GPU_SMOKE_TESTS="${ENABLE_GPU}"
    -DENABLE_ONEDNN_CONV_BACKEND="${ENABLE_ONEDNN}"
    -DENABLE_ONEDNN_PARITY_PROBE=ON
    -DENABLE_VTUNE_MARKERS=OFF
)

if [[ -n "${GENERATOR}" ]]; then
    CMAKE_ARGS=(-G "${GENERATOR}" "${CMAKE_ARGS[@]}")
elif command -v ninja >/dev/null 2>&1; then
    CMAKE_ARGS=(-G Ninja "${CMAKE_ARGS[@]}")
fi

run cmake "${CMAKE_ARGS[@]}"

QUICK_TARGETS=(
    ppn_train
    test_matrix
    test_gemm_microkernels
    test_gemm_gotoblas_driver
    test_activations
    test_linear
    test_mlp
    test_autodiff
    test_gradient_check
    test_autograd_engine
    test_cnn_autograd
    test_resnet_network
    test_distributed_sync
    test_optimizer
    test_tiny_imagenet_dataset
    test_ist_runtime
    test_init
    test_cnn_configurable
    test_conv2d_onednn_backend
)

if [[ "${RUN_ONEDNN_PARITY}" != "0" ]]; then
    QUICK_TARGETS+=(test_onednn_conv_parity_probe)
fi

INCLUDE_DATA_TESTS=0
if [[ "${RUN_DATA_TESTS}" == "1" ]] || [[ "${RUN_DATA_TESTS}" == "auto" && "$(have_mnist && echo yes || echo no)" == "yes" ]]; then
    INCLUDE_DATA_TESTS=1
    QUICK_TARGETS+=(test_dataloader)
    DATA_REGEX="|DataLoaderTest|CLICNNSmokeTest"
    log "MNIST data found: data-dependent tests will be included."
else
    DATA_REGEX=""
    log "MNIST data not found or disabled: data-dependent tests will be skipped."
fi

if [[ "${PROFILE}" == "full" ]]; then
    run cmake --build "${BUILD_DIR}" --parallel 1
    CTEST_REGEX=".*"
else
    run cmake --build "${BUILD_DIR}" --parallel 1 --target "${QUICK_TARGETS[@]}"
    CTEST_REGEX="MatrixTest|GemmMicrokernelsTest|GemmGotoBlasDriverTest|GemmGotoBlasAvx512DriverTest|ActivationsTest|LinearTest|MLPTest|AutodiffTest|GradientCheckTest|AutogradEngineTest|CNNAutogradTest|ResNetNetworkTest|DistributedSyncTest|OptimizerTest|TinyImageNetDatasetTest|IstRuntimeTest|InitTest|CNNConfigurableTest|Conv2DOneDnnBackendTest|CLI.*${DATA_REGEX}"
fi

CTEST_ARGS=(
    --test-dir "${BUILD_DIR}"
    --output-on-failure
    --parallel 1
    --timeout 120
    -R "${CTEST_REGEX}"
)

if [[ "${INCLUDE_DATA_TESTS}" == "0" ]]; then
    CTEST_ARGS+=(-E "DataLoaderTest|CLICNNSmokeTest")
fi

run ctest "${CTEST_ARGS[@]}"

if [[ "${RUN_ONEDNN_PARITY}" != "0" && -x "${BUILD_DIR}/test_onednn_conv_parity_probe" ]]; then
    log "Running oneDNN conv parity probe. Missing oneDNN is reported as a skip by the probe."
    run "${BUILD_DIR}/test_onednn_conv_parity_probe"
fi

if have_mnist; then
    SMOKE_OUT_DIR="${LOG_ROOT}/train_mnist_mlp_epochs0"
    log "Running one tiny positive ppn_train pass with MNIST and epochs=0."
    run "${BUILD_DIR}/ppn_train" \
        --dataset mnist \
        --data_dir "${PROJECT_ROOT}/mnist" \
        --model mlp \
        --epochs 0 \
        --batch_size 10000 \
        --hidden_size 16 \
        --seed 42 \
        --out_dir "${SMOKE_OUT_DIR}"
    expect_file "${SMOKE_OUT_DIR}/metrics.csv"
    expect_file "${SMOKE_OUT_DIR}/profile_run_summary.csv"

    if [[ "${RUN_CHECKPOINT}" == "1" || "${RUN_CHECKPOINT}" == "auto" ]]; then
        CHECKPOINT_OUT_DIR="${LOG_ROOT}/train_mnist_checkpoint"
        CHECKPOINT_PATH="${CHECKPOINT_OUT_DIR}/smoke.ckpt"
        log "Running checkpoint save/resume smoke with a tiny one-epoch MLP."
        run "${BUILD_DIR}/ppn_train" \
            --dataset mnist \
            --data_dir "${PROJECT_ROOT}/mnist" \
            --model mlp \
            --epochs 1 \
            --batch_size 10000 \
            --hidden_size 8 \
            --optimizer sgd \
            --learning_rate 0.001 \
            --seed 42 \
            --save_checkpoint "${CHECKPOINT_PATH}" \
            --out_dir "${CHECKPOINT_OUT_DIR}/save"
        expect_file "${CHECKPOINT_PATH}"
        run "${BUILD_DIR}/ppn_train" \
            --resume "${CHECKPOINT_PATH}" \
            --epochs 0 \
            --data_dir "${PROJECT_ROOT}/mnist" \
            --out_dir "${CHECKPOINT_OUT_DIR}/resume"
        expect_file "${CHECKPOINT_OUT_DIR}/resume/metrics.csv"
    fi
else
    log "Skipping positive ppn_train pass because MNIST files are not present in ${PROJECT_ROOT}/mnist."
fi

if [[ "${RUN_GPU}" == "1" ]]; then
    if [[ "${ENABLE_GPU}" != "ON" ]]; then
        log "GPU smoke requested, but PPN_ENABLE_GPU_SMOKE_TESTS is not ON; skipping GPU targets."
    else
        log "Running optional CUDA/cuDNN smoke targets."
        run cmake --build "${BUILD_DIR}" --parallel 1 --target cuda_runtime_smoke_test
        run "${BUILD_DIR}/cuda_runtime_smoke_test"
        if cmake --build "${BUILD_DIR}" --parallel 1 --target cudnn_smoke_test; then
            run "${BUILD_DIR}/cudnn_smoke_test"
        else
            log "cuDNN smoke target is unavailable in this build; CUDA runtime smoke completed."
        fi
    fi
fi

if [[ "${RUN_MPI}" == "1" ]]; then
    log "MPI smoke requested. Reconfigure/build with ENABLE_MPI=ON in a separate directory."
    MPI_BUILD_DIR="${PPN_MPI_BUILD_DIR:-${PROJECT_ROOT}/build-zen-smoke-mpi}"
    run cmake \
        -S "${PROJECT_ROOT}" \
        -B "${MPI_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_MPI=ON \
        -DENABLE_GPU_SMOKE_TESTS=OFF \
        -DENABLE_ONEDNN_CONV_BACKEND="${ENABLE_ONEDNN}" \
        -DENABLE_ONEDNN_PARITY_PROBE=ON \
        -DENABLE_VTUNE_MARKERS=OFF
    run cmake --build "${MPI_BUILD_DIR}" --parallel 1 --target test_distributed_sync_mpi_contracts
    run ctest --test-dir "${MPI_BUILD_DIR}" --output-on-failure --parallel 1 --timeout 120 -R DistributedSyncMPIContractsWorldSize2
else
    log "MPI smoke skipped. For a later two-task test, submit with PPN_RUN_MPI=1 and request enough Slurm tasks."
fi

log "Smoke test finished successfully."
echo "Logs are in: ${LOG_ROOT}"

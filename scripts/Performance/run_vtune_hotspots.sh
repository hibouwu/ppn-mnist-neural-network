#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-vtune}"
COLLECTOR="${VTUNE_COLLECTOR:-hotspots}"
RESULT_DIR="${VTUNE_RESULT_DIR:-output/vtune/${COLLECTOR}_$(date +%Y%m%d_%H%M%S)}"

if ! command -v vtune >/dev/null 2>&1; then
    echo "error: 'vtune' command not found."
    echo "hint: source your oneAPI environment first, for example:"
    echo "  source /opt/intel/oneapi/setvars.sh"
    exit 1
fi

cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DENABLE_VTUNE_MARKERS=ON
cmake --build "${BUILD_DIR}" -j"$(nproc)" --target ppn_train

if [ "$#" -eq 0 ]; then
    set -- --epochs 1 --batch_size 256 --hidden_size 128 --data_dir mnist
fi

mkdir -p "$(dirname "${RESULT_DIR}")"

vtune -collect "${COLLECTOR}" \
    -result-dir "${RESULT_DIR}" \
    -- "${BUILD_DIR}/ppn_train" "$@"

echo "VTune result saved to ${RESULT_DIR}"

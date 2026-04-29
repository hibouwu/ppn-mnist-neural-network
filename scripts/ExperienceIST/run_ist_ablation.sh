#!/usr/bin/env bash
set -euo pipefail

# IST vs Dense ablation runner (matrix mode + dataset presets).
#
# Defaults preserve old behavior (MNIST ablation):
# - 1 seed: 42
# - 1 lr: 0.01
# - runs dense_k1 + ist_k1/2/4
#
# Examples:
#   bash scripts/ExperienceIST/run_ist_ablation.sh
#
#   SEED_LIST=42,43,44 LR_LIST=0.01,0.005 EPOCHS=10 \
#   bash scripts/ExperienceIST/run_ist_ablation.sh
#
#   MODE=tiny_smoke DATA_DIR=tiny-imagenet-200 \
#   bash scripts/ExperienceIST/run_ist_ablation.sh
#
#   MODE=tiny_compare DATA_DIR=tiny-imagenet-200 \
#   bash scripts/ExperienceIST/run_ist_ablation.sh

BUILD="${BUILD:-./build-mpi/ppn_train}"
WORLD_SIZE="${WORLD_SIZE:-2}"
DATASET="${DATASET:-mnist}"
DATA_DIR="${DATA_DIR:-mnist}"
MODEL="${MODEL:-cnn}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OPTIMIZER="${OPTIMIZER:-momentum_sgd}"
MOMENTUM="${MOMENTUM:-0.9}"
OMP_THREADS="${OMP_THREADS:-4}"
OUT_ROOT="${OUT_ROOT:-output/IST_ablation_$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-mnist_ablation}"    # mnist_ablation / tiny_smoke / tiny_compare

# Matrix controls:
SEED_LIST="${SEED_LIST:-}"         # comma-separated, e.g. 42,43,44
LR_LIST="${LR_LIST:-}"             # comma-separated, e.g. 0.01,0.005
IST_K_LIST="${IST_K_LIST:-}"       # comma-separated
RUN_DENSE="${RUN_DENSE:-1}"        # 1/0
RUN_IST="${RUN_IST:-1}"            # 1/0

export OMP_NUM_THREADS="${OMP_THREADS}"

if [[ ! -x "${BUILD}" ]]; then
  echo "Error: executable not found or not executable: ${BUILD}" >&2
  echo "Hint: cmake --build build-mpi -j\$(nproc) --target ppn_train" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

apply_mode_defaults() {
  case "${MODE}" in
    mnist_ablation)
      SEED_LIST="${SEED_LIST:-42}"
      LR_LIST="${LR_LIST:-0.01}"
      IST_K_LIST="${IST_K_LIST:-1,2,4}"
      ;;
    tiny_smoke)
      DATASET="${DATASET:-tiny-imagenet}"
      DATASET="tiny-imagenet"
      BATCH_SIZE="${BATCH_SIZE:-64}"
      EPOCHS="${EPOCHS:-1}"
      SEED_LIST="${SEED_LIST:-42}"
      LR_LIST="${LR_LIST:-0.01}"
      IST_K_LIST="${IST_K_LIST:-4}"
      ;;
    tiny_compare)
      DATASET="${DATASET:-tiny-imagenet}"
      DATASET="tiny-imagenet"
      BATCH_SIZE="${BATCH_SIZE:-64}"
      EPOCHS="${EPOCHS:-3}"
      SEED_LIST="${SEED_LIST:-42,43}"
      LR_LIST="${LR_LIST:-0.01}"
      IST_K_LIST="${IST_K_LIST:-4}"
      ;;
    *)
      echo "Error: unsupported MODE=${MODE}. Expected mnist_ablation|tiny_smoke|tiny_compare" >&2
      exit 1
      ;;
  esac
}

split_csv() {
  local s="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "${s}"
}

sanitize_token() {
  # Replace dot for file-system-friendly names
  echo "$1" | tr '.' 'p'
}

run_exp() {
  local tag="$1"
  local mode="$2"
  local k="$3"
  local lr="$4"
  local seed="$5"
  local lr_token
  lr_token="$(sanitize_token "${lr}")"
  local out_dir="${OUT_ROOT}/${tag}_lr${lr_token}_s${seed}"
  mkdir -p "${out_dir}"

  echo "===== RUN ${tag} (mode=${mode}, k=${k}, lr=${lr}, seed=${seed}) ====="
  mpiexec -n "${WORLD_SIZE}" "${BUILD}" \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${lr}" \
    --optimizer "${OPTIMIZER}" \
    --momentum "${MOMENTUM}" \
    --train_mode "${mode}" \
    --ist_local_steps "${k}" \
    --seed "${seed}" \
    --out_dir "${out_dir}" \
    2>&1 | tee "${out_dir}/train.log"
}

extract_field() {
  local line="$1"
  local expr="$2"
  echo "${line}" | sed -n "${expr}"
}

generate_summary_all() {
  local summary_csv="${OUT_ROOT}/summary_all.csv"
  {
    echo "tag,lr,seed,train_loss,train_acc,test_loss,test_acc,epoch_s,sync_total_s"
    for d in "${OUT_ROOT}"/*; do
      [[ -d "${d}" ]] || continue
      local base
      base="$(basename "${d}")"
      local log_file="${d}/train.log"
      [[ -f "${log_file}" ]] || continue

      local tag
      tag="$(echo "${base}" | sed -n 's/_lr.*//p')"
      local lr_token
      lr_token="$(echo "${base}" | sed -n 's/.*_lr\([^_]*\)_s.*/\1/p')"
      local lr
      lr="$(echo "${lr_token}" | tr 'p' '.')"
      local seed
      seed="$(echo "${base}" | sed -n 's/.*_s\([0-9]*\)$/\1/p')"

      local line
      line="$(grep "Epoch " "${log_file}" | tail -n1 || true)"
      if [[ -z "${line}" ]]; then
        echo "${tag},${lr},${seed},N/A,N/A,N/A,N/A,N/A,N/A"
        continue
      fi

      local train_loss train_acc test_loss test_acc epoch_s sync_s
      train_loss="$(extract_field "${line}" 's/.*\[Train\] loss = \([0-9.]*\),.*/\1/p')"
      train_acc="$(extract_field "${line}" 's/.*\[Train\] loss = [0-9.]*, acc = \([0-9.]*\)%.*/\1/p')"
      test_loss="$(extract_field "${line}" 's/.*\[Test\] loss = \([0-9.]*\),.*/\1/p')"
      test_acc="$(extract_field "${line}" 's/.*\[Test\] loss = [0-9.]*, acc = \([0-9.]*\)%.*/\1/p')"
      epoch_s="$(extract_field "${line}" 's/.*| epoch = \([0-9.]*\)s,.*/\1/p')"
      sync_s="$(extract_field "${line}" 's/.*sync_total = \([0-9.]*\)s,.*/\1/p')"

      echo "${tag},${lr},${seed},${train_loss},${train_acc},${test_loss},${test_acc},${epoch_s},${sync_s}"
    done
  } > "${summary_csv}"
  echo "Summary written to: ${summary_csv}"
}

generate_grouped_means() {
  local summary_csv="${OUT_ROOT}/summary_all.csv"
  local means_csv="${OUT_ROOT}/summary_grouped_means.csv"
  python3 - "$summary_csv" "$means_csv" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

summary_csv, means_csv = sys.argv[1], sys.argv[2]
group = defaultdict(lambda: {"acc": [], "epoch": [], "sync": []})

with open(summary_csv, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            key = (row["tag"], row["lr"])
            group[key]["acc"].append(float(row["test_acc"]))
            group[key]["epoch"].append(float(row["epoch_s"]))
            group[key]["sync"].append(float(row["sync_total_s"]))
        except Exception:
            continue

with open(means_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tag", "lr", "mean_test_acc", "mean_epoch_s", "mean_sync_total_s", "n"])
    for (tag, lr) in sorted(group.keys()):
        vals = group[(tag, lr)]
        w.writerow([
            tag,
            lr,
            f"{statistics.mean(vals['acc']):.4f}",
            f"{statistics.mean(vals['epoch']):.4f}",
            f"{statistics.mean(vals['sync']):.4f}",
            len(vals["acc"]),
        ])
PY
  echo "Grouped means written to: ${means_csv}"
}

apply_mode_defaults
split_csv "${SEED_LIST}" SEEDS
split_csv "${LR_LIST}" LRS
split_csv "${IST_K_LIST}" IST_KS

echo "MODE=${MODE}"
echo "BUILD=${BUILD}"
echo "DATASET=${DATASET}"
echo "DATA_DIR=${DATA_DIR}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "SEED_LIST=${SEED_LIST}"
echo "LR_LIST=${LR_LIST}"
echo "IST_K_LIST=${IST_K_LIST}"
echo "OUT_ROOT=${OUT_ROOT}"
echo

for lr in "${LRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [[ "${RUN_DENSE}" == "1" ]]; then
      run_exp "dense_k1" "dense" "1" "${lr}" "${seed}"
    fi
    if [[ "${RUN_IST}" == "1" ]]; then
      for k in "${IST_KS[@]}"; do
        run_exp "ist_k${k}" "ist" "${k}" "${lr}" "${seed}"
      done
    fi
  done
done

generate_summary_all
generate_grouped_means
echo
cat "${OUT_ROOT}/summary_all.csv"
echo
cat "${OUT_ROOT}/summary_grouped_means.csv"

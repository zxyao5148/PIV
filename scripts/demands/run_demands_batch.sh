#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
LOG_DIR="${ROOT_DIR}/logs/run_demands"
CHECKPOINT_DIR="${ROOT_DIR}/outputs/checkpoints/demands"
OUT_FIG="${ROOT_DIR}/outputs/figs/demands"
SEEDS_CSV="$(seq -s, 1 20)"
RHOS_CSV="0.5"
GPU_IDS_CSV="1,2,3"
WORKERS_PER_GPU=2
DEVICE_MODE="auto"
SUM_RHO=""
SUM_RHO_LIST_CSV=""
SUM_RHO_SEEDS=()
RUN_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/demands/run_demands_batch.sh [options] [-- extra train_demands args]

Options:
  --seeds CSV         Comma-separated seeds to run. Default: 1,2,...,100
  --rhos CSV          Comma-separated rho values to run. Default: 0.5
  --gpu-ids CSV       Physical CUDA GPU ids for training workers. Default: 1,2,3
  --workers-per-gpu N Concurrent workers per GPU. Default: 2
  --device MODE       `auto`, `cuda`, or `cpu`. Default: auto
  --log-dir PATH      Log directory. Default: logs/run_demands
  --checkpoint-dir P  Checkpoint directory. Default: outputs/checkpoints/demands
  --out-fig PATH      Aggregated figure directory. Default: outputs/figs/demands
  --python BIN        Python executable. Default: $PYTHON or python
  --sum-rho X         Single rho for per-rho figures in test_demands.py
  --sum-rho-list CSV  Comma-separated rhos for the cross-rho MSE heatmap
  --sum-rho-seeds X   Repeated rho:seed1,seed2 spec forwarded to test_demands.py
  --help              Show this help

Examples:
  scripts/demands/run_demands_batch.sh --seeds 1,2,3 --rhos 0.1,0.5,0.9
  scripts/demands/run_demands_batch.sh \
    --seeds 1,2,3,4 --rhos 0.1,0.5 \
    --checkpoint-dir outputs/checkpoints/demands_example \
    --sum-rho 0.5 --sum-rho-list 0.1,0.5 \
    --sum-rho-seeds 0.1:1,2 --sum-rho-seeds 0.5:3,4 \
    -- --methods PIV TrueIV NoneIV --downstream deepiv dfiv
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      SEEDS_CSV="$2"
      shift 2
      ;;
    --rhos)
      RHOS_CSV="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS_CSV="$2"
      shift 2
      ;;
    --workers-per-gpu)
      WORKERS_PER_GPU="$2"
      shift 2
      ;;
    --device)
      DEVICE_MODE="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --out-fig)
      OUT_FIG="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --sum-rho)
      SUM_RHO="$2"
      shift 2
      ;;
    --sum-rho-list)
      SUM_RHO_LIST_CSV="$2"
      shift 2
      ;;
    --sum-rho-seeds)
      SUM_RHO_SEEDS+=("$2")
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      RUN_ARGS+=("$@")
      break
      ;;
    *)
      RUN_ARGS+=("$1")
      shift
      ;;
  esac
done

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a RHOS <<< "${RHOS_CSV}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"

mkdir -p "${LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

resolve_device() {
  local requested="$1"
  if [[ "${requested}" == "cpu" || "${requested}" == "cuda" ]]; then
    echo "${requested}"
    return 0
  fi
  "${PYTHON_BIN}" - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
}

DEVICE="$(
  resolve_device "${DEVICE_MODE}"
)"

if [[ "${DEVICE}" == "cuda" && "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "No GPU ids provided via --gpu-ids" >&2
  exit 1
fi

JOB_FILE="$(mktemp "${TMPDIR:-/tmp}/run_demands_jobs.XXXXXX")"
LOCK_FILE="$(mktemp "${TMPDIR:-/tmp}/run_demands_lock.XXXXXX")"
cleanup() {
  rm -f "${JOB_FILE}" "${LOCK_FILE}"
}
trap cleanup EXIT

for rho in "${RHOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    printf '%s %s\n' "${rho}" "${seed}" >> "${JOB_FILE}"
  done
done

pop_next_job() {
  JOB_FILE_PATH="${JOB_FILE}" LOCK_FILE_PATH="${LOCK_FILE}" "${PYTHON_BIN}" - <<'PY'
import fcntl
import os

job_file = os.environ["JOB_FILE_PATH"]
lock_file = os.environ["LOCK_FILE_PATH"]

with open(lock_file, "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not os.path.exists(job_file):
        print("")
    else:
        with open(job_file, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        if not lines:
            print("")
        else:
            first = lines[0].strip()
            with open(job_file, "w", encoding="utf-8") as fh:
                fh.writelines(lines[1:])
            print(first)
PY
}

run_worker() {
  local gpu_id="${1:-}"
  local job
  local rho
  local seed
  local rho_tag
  local log_path

  while true; do
    job="$(pop_next_job)"
    if [[ -z "${job}" ]]; then
      echo "=== worker gpu=${gpu_id}: queue empty"
      break
    fi

    read -r rho seed <<< "${job}"
    rho_tag="${rho//./p}"
    rho_tag="${rho_tag//-/m}"
    log_path="${LOG_DIR}/run_demands_rho${rho_tag}_seed${seed}.log"
    echo "=== worker device=${DEVICE}${gpu_id:+ gpu=${gpu_id}}: train_demands rho=${rho} seed=${seed} -> ${log_path}"

    if [[ "${DEVICE}" == "cuda" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/demands/train_demands.py" \
        --seed "${seed}" \
        --rho "${rho}" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        "${RUN_ARGS[@]}" \
        --device cuda \
        2>&1 | tee "${log_path}"
    else
      "${PYTHON_BIN}" "${ROOT_DIR}/scripts/demands/train_demands.py" \
        --seed "${seed}" \
        --rho "${rho}" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        "${RUN_ARGS[@]}" \
        --device cpu \
        2>&1 | tee "${log_path}"
    fi
  done
}

WORKER_PIDS=()
if [[ "${DEVICE}" == "cuda" ]]; then
  for gpu_id in "${GPU_IDS[@]}"; do
    for (( w=0; w<WORKERS_PER_GPU; w++ )); do
      run_worker "${gpu_id}" &
      WORKER_PIDS+=("$!")
    done
  done
else
  run_worker &
  WORKER_PIDS+=("$!")
fi

for pid in "${WORKER_PIDS[@]}"; do
  wait "${pid}"
done

SUM_ARGS=(
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --out-fig "${OUT_FIG}"
)
SUM_RHO_SEED_COUNT=0
if [[ ${#SUM_RHO_SEEDS[@]:-} -gt 0 ]]; then
  SUM_RHO_SEED_COUNT=${#SUM_RHO_SEEDS[@]}
fi
if [[ -z "${SUM_RHO}" && -z "${SUM_RHO_LIST_CSV}" && "${SUM_RHO_SEED_COUNT}" -eq 0 ]]; then
  # Default summary scope to the same rhos we just trained so the
  # aggregator always has an explicit rho selection.
  SUM_RHO_LIST_CSV="${RHOS_CSV}"
fi
if [[ -n "${SUM_RHO}" ]]; then
  SUM_ARGS+=(--rho "${SUM_RHO}")
fi
if [[ -n "${SUM_RHO_LIST_CSV}" ]]; then
  IFS=',' read -r -a SUM_RHO_LIST <<< "${SUM_RHO_LIST_CSV}"
  SUM_ARGS+=(--rho-list "${SUM_RHO_LIST[@]}")
fi
if [[ "${SUM_RHO_SEED_COUNT}" -gt 0 ]]; then
  for spec in "${SUM_RHO_SEEDS[@]}"; do
    SUM_ARGS+=(--rho-seeds "${spec}")
  done
fi
SUM_ARGS+=("${RUN_ARGS[@]}")

echo "=== test_demands -> ${OUT_FIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/demands/test_demands.py" "${SUM_ARGS[@]}"

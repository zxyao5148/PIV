#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
LOG_DIR="${ROOT_DIR}/logs/run_ibm"
CHECKPOINT_DIR="${ROOT_DIR}/outputs/checkpoints/ibm"
TRAIN_SEEDS_CSV="$(seq -s, 1 10)"
DATA_SEED=1
AUTOENC_SEED=1
TREATMENT_TYPES_CSV="continuous,binary"
SURFACES_CSV=""
DA_VALUES_CSV="1"
DESIGN_SEED=1
GPU_IDS_CSV="0"
WORKERS_PER_GPU=2
DEVICE_MODE="auto"
RUN_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/ibm/run_ibm_batch.sh [options] [-- extra train_ibm args]

Options:
  --seeds CSV                 Comma-separated train seeds. Default: 1,2,3,4,5,6,7,8,9,10
  --data-seed N               Fixed data seed. Default: 1
  --autoenc-seed N            Fixed autoencoder seed. Default: 1
  --treatment-types CSV       Comma-separated treatment types (continuous,binary). Default: both
  --surfaces CSV              Comma-separated surfaces to run. When omitted, runs all surfaces
                              compatible with each treatment type. Ignored if extra args include --surface.
  --da-values CSV             Comma-separated dA values. Default: 1
  --design-seed N             Fixed design seed. Default: 1
  --gpu-ids CSV               Physical CUDA GPU ids. Default: 0
  --workers-per-gpu N         Concurrent workers per GPU. Default: 2
  --device MODE               auto, cuda, or cpu. Default: auto
  --log-dir PATH              Log directory. Default: logs/run_ibm
  --checkpoint-dir PATH       Checkpoint directory. Default: outputs/checkpoints/ibm
  --python BIN                Python executable. Default: $PYTHON or python
  --help                      Show this help

Examples:
  # Quick smoke test: 2 seeds, continuous only, TrueIV only, no downstream
  scripts/ibm/run_ibm_batch.sh \
    --seeds 1,2 --treatment-types continuous --device cpu \
    -- --surface poly --methods TrueIV --skip-downstream \
       --n 128 --vae-epochs 1 --resid-epochs 1 --probe-epochs 1

  # Full-ish run: 10 seeds, scalar treatment, selected methods/downstream
  scripts/ibm/run_ibm_batch.sh \
    --seeds 1,2,3,4,5,6,7,8,9,10 \
    -- --methods PIV Proxy-Orth TrueIV NoneIV --downstream deepiv dfiv
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds|--train-seeds)
      TRAIN_SEEDS_CSV="$2"; shift 2 ;;
    --data-seed)
      DATA_SEED="$2"; shift 2 ;;
    --autoenc-seed)
      AUTOENC_SEED="$2"; shift 2 ;;
    --da-values)
      DA_VALUES_CSV="$2"; shift 2 ;;
    --treatment-types)
      TREATMENT_TYPES_CSV="$2"; shift 2 ;;
    --surfaces)
      SURFACES_CSV="$2"; shift 2 ;;
    --design-seed)
      DESIGN_SEED="$2"; shift 2 ;;
    --gpu-ids)
      GPU_IDS_CSV="$2"; shift 2 ;;
    --workers-per-gpu)
      WORKERS_PER_GPU="$2"; shift 2 ;;
    --device)
      DEVICE_MODE="$2"; shift 2 ;;
    --log-dir)
      LOG_DIR="$2"; shift 2 ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    --)
      shift; RUN_ARGS+=("$@"); break ;;
    *)
      RUN_ARGS+=("$1"); shift ;;
  esac
done

IFS=',' read -r -a TRAIN_SEEDS <<< "${TRAIN_SEEDS_CSV}"
IFS=',' read -r -a TREATMENT_TYPES <<< "${TREATMENT_TYPES_CSV}"
IFS=',' read -r -a REQUESTED_SURFACES <<< "${SURFACES_CSV}"
IFS=',' read -r -a DA_VALUES <<< "${DA_VALUES_CSV}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"

get_surfaces_for_type() {
  case "$1" in
    continuous) echo "poly varying_slope soft_threshold sinusoidal" ;;
    binary)     echo "binary_linear binary_threshold_modulated" ;;
    *)          echo "" ;;
  esac
}

surface_allowed_for_type() {
  local treatment_type="$1"
  local surface="$2"
  case "${treatment_type}:${surface}" in
    continuous:poly|continuous:varying_slope|continuous:soft_threshold|continuous:sinusoidal) return 0 ;;
    binary:binary_linear|binary:binary_threshold_modulated) return 0 ;;
    *) return 1 ;;
  esac
}

# If RUN_ARGS contains --surface, honour it (single surface mode).
# Otherwise expand all surfaces for each treatment type.
EXPLICIT_SURFACE=""
for ((i=0; i<${#RUN_ARGS[@]}; i++)); do
  if [[ "${RUN_ARGS[$i]}" == "--surface" ]]; then
    EXPLICIT_SURFACE="${RUN_ARGS[$((i+1))]}"
    break
  fi
done

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

resolve_device() {
  local requested="$1"
  if [[ "${requested}" == "cpu" || "${requested}" == "cuda" ]]; then
    echo "${requested}"; return 0
  fi
  "${PYTHON_BIN}" - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
}

DEVICE="$(resolve_device "${DEVICE_MODE}")"

if [[ "${DEVICE}" == "cuda" && "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "No GPU ids provided via --gpu-ids" >&2; exit 1
fi

JOB_FILE="$(mktemp "${TMPDIR:-/tmp}/run_ibm_jobs.XXXXXX")"
LOCK_FILE="$(mktemp "${TMPDIR:-/tmp}/run_ibm_lock.XXXXXX")"
cleanup() { rm -f "${JOB_FILE}" "${LOCK_FILE}"; }
trap cleanup EXIT

for train_seed in "${TRAIN_SEEDS[@]}"; do
  for treatment_type in "${TREATMENT_TYPES[@]}"; do
    if [[ -n "${EXPLICIT_SURFACE}" ]]; then
      surfaces=("${EXPLICIT_SURFACE}")
    elif [[ -n "${SURFACES_CSV}" ]]; then
      surfaces=()
      for surface in "${REQUESTED_SURFACES[@]}"; do
        if surface_allowed_for_type "${treatment_type}" "${surface}"; then
          surfaces+=("${surface}")
        fi
      done
    else
      IFS=' ' read -r -a surfaces <<< "$(get_surfaces_for_type "${treatment_type}")"
    fi
    for surface in "${surfaces[@]}"; do
      for dA in "${DA_VALUES[@]}"; do
        printf '%s %s %s %s\n' "${treatment_type}" "${surface}" "${dA}" "${train_seed}" >> "${JOB_FILE}"
      done
    done
  done
done

pop_next_job() {
  JOB_FILE_PATH="${JOB_FILE}" LOCK_FILE_PATH="${LOCK_FILE}" "${PYTHON_BIN}" - <<'PY'
import fcntl, os
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
  local job treatment_type surface dA train_seed log_path da_tag

  while true; do
    job="$(pop_next_job)"
    if [[ -z "${job}" ]]; then
      echo "=== worker gpu=${gpu_id}: queue empty"
      break
    fi

    read -r treatment_type surface dA train_seed <<< "${job}"
    da_tag="$( [[ "${dA}" -gt 1 ]] && echo "_dA${dA}" || echo "" )"
    log_path="${LOG_DIR}/run_ibm_${treatment_type}_${surface}${da_tag}_data${DATA_SEED}_ae${AUTOENC_SEED}_train${train_seed}.log"
    echo "=== worker device=${DEVICE}${gpu_id:+ gpu=${gpu_id}}: treatment=${treatment_type} surface=${surface} dA=${dA} data_seed=${DATA_SEED} autoenc_seed=${AUTOENC_SEED} train_seed=${train_seed} -> ${log_path}"

    local base_args=(
      "--treatment-type" "${treatment_type}"
      "--design-seed" "${DESIGN_SEED}"
      "--data-seed" "${DATA_SEED}"
      "--autoenc-seed" "${AUTOENC_SEED}"
      "--train-seed" "${train_seed}"
      "--dA" "${dA}"
      "--checkpoint-dir" "${CHECKPOINT_DIR}"
    )

    # Only add --surface if not already in RUN_ARGS
    if [[ -z "${EXPLICIT_SURFACE}" ]]; then
      base_args+=("--surface" "${surface}")
    fi

    if [[ "${DEVICE}" == "cuda" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
        PYTHONUNBUFFERED=1 \
        "${PYTHON_BIN}" -u "${ROOT_DIR}/scripts/ibm/train_ibm.py" \
        "${base_args[@]}" "${RUN_ARGS[@]}" --device cuda \
        2>&1 | tee "${log_path}"
    else
      PYTHONUNBUFFERED=1 \
        "${PYTHON_BIN}" -u "${ROOT_DIR}/scripts/ibm/train_ibm.py" \
        "${base_args[@]}" "${RUN_ARGS[@]}" --device cpu \
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
  run_worker "" &
  WORKER_PIDS+=("$!")
fi

for pid in "${WORKER_PIDS[@]}"; do
  wait "${pid}"
done

echo "=== All IBM training jobs complete."

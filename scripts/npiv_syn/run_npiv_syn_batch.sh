#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
LOG_DIR="${ROOT_DIR}/logs/run_npiv_syn"
CHECKPOINT_DIR="${ROOT_DIR}/outputs/checkpoints/npiv_syn"
OUT_FIG="${ROOT_DIR}/outputs/figs/npiv_syn"
TRAIN_SEEDS_CSV="$(seq -s, 1 5)"
TREATMENT_TYPES_CSV="continuous,binary"
DESIGN_SEED=1
DATA_SEED=1
GPU_IDS_CSV="0"
WORKERS_PER_GPU=2
DEVICE_MODE="auto"
SUM_TREATMENT_TYPES_CSV=""
SUM_SURFACES_CSV=""
RUN_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/npiv_syn/run_npiv_syn_batch.sh [options] [-- extra train_npiv_syn args]

Options:
  --train-seeds CSV       Comma-separated train seeds. Default: 1,2,3,4,5
  --treatment-types CSV   Comma-separated treatment types (continuous,binary). Default: both
  --design-seed N         Fixed design seed. Default: 1
  --data-seed N           Fixed data seed. Default: 1
  --gpu-ids CSV           Physical CUDA GPU ids. Default: 0
  --workers-per-gpu N     Concurrent workers per GPU. Default: 2
  --device MODE           auto, cuda, or cpu. Default: auto
  --log-dir PATH          Log directory. Default: logs/run_npiv_syn
  --checkpoint-dir PATH   Checkpoint directory. Default: outputs/checkpoints/npiv_syn
  --out-fig PATH          Figure output directory. Default: outputs/figs/npiv_syn
  --python BIN            Python executable. Default: $PYTHON or python
  --sum-treatment-types CSV  Treatment types for test_npiv_syn.py aggregation
  --sum-surfaces CSV      Surfaces for test_npiv_syn.py aggregation
  --help                  Show this help

Examples:
  # Quick smoke test: 2 seeds, continuous only, poly surface, TrueIV only
  scripts/npiv_syn/run_npiv_syn_batch.sh \
    --train-seeds 1,2 --treatment-types continuous \
    -- --surface poly --methods TrueIV --downstream deepiv

  # Full run: 5 seeds, all surfaces and treatment types
  scripts/npiv_syn/run_npiv_syn_batch.sh \
    --train-seeds 1,2,3,4,5 \
    -- --methods PIV TrueIV NoneIV --downstream deepiv dfiv
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-seeds)
      TRAIN_SEEDS_CSV="$2"; shift 2 ;;
    --treatment-types)
      TREATMENT_TYPES_CSV="$2"; shift 2 ;;
    --design-seed)
      DESIGN_SEED="$2"; shift 2 ;;
    --data-seed)
      DATA_SEED="$2"; shift 2 ;;
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
    --out-fig)
      OUT_FIG="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --sum-treatment-types)
      SUM_TREATMENT_TYPES_CSV="$2"; shift 2 ;;
    --sum-surfaces)
      SUM_SURFACES_CSV="$2"; shift 2 ;;
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
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"

declare -A SURFACES_FOR_TYPE
SURFACES_FOR_TYPE["continuous"]="poly tanh thresh varying_slope"
SURFACES_FOR_TYPE["binary"]="binary_linear binary_threshold_modulated"

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

JOB_FILE="$(mktemp "${TMPDIR:-/tmp}/run_npiv_syn_jobs.XXXXXX")"
LOCK_FILE="$(mktemp "${TMPDIR:-/tmp}/run_npiv_syn_lock.XXXXXX")"
cleanup() { rm -f "${JOB_FILE}" "${LOCK_FILE}"; }
trap cleanup EXIT

for treatment_type in "${TREATMENT_TYPES[@]}"; do
  if [[ -n "${EXPLICIT_SURFACE}" ]]; then
    surfaces=("${EXPLICIT_SURFACE}")
  else
    IFS=' ' read -r -a surfaces <<< "${SURFACES_FOR_TYPE[$treatment_type]:-}"
  fi
  for surface in "${surfaces[@]}"; do
    for train_seed in "${TRAIN_SEEDS[@]}"; do
      printf '%s %s %s\n' "${treatment_type}" "${surface}" "${train_seed}" >> "${JOB_FILE}"
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
  local job treatment_type surface train_seed log_path

  while true; do
    job="$(pop_next_job)"
    if [[ -z "${job}" ]]; then
      echo "=== worker gpu=${gpu_id}: queue empty"
      break
    fi

    read -r treatment_type surface train_seed <<< "${job}"
    log_path="${LOG_DIR}/run_npiv_syn_${treatment_type}_${surface}_train${train_seed}.log"
    echo "=== worker device=${DEVICE}${gpu_id:+ gpu=${gpu_id}}: treatment=${treatment_type} surface=${surface} train_seed=${train_seed} -> ${log_path}"

    local base_args=(
      "--treatment-type" "${treatment_type}"
      "--design-seed" "${DESIGN_SEED}"
      "--data-seed" "${DATA_SEED}"
      "--train-seed" "${train_seed}"
      "--checkpoint-dir" "${CHECKPOINT_DIR}"
    )

    # Only add --surface if not already in RUN_ARGS
    if [[ -z "${EXPLICIT_SURFACE}" ]]; then
      base_args+=("--surface" "${surface}")
    fi

    if [[ "${DEVICE}" == "cuda" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/npiv_syn/train_npiv_syn.py" \
        "${base_args[@]}" "${RUN_ARGS[@]}" --device cuda \
        2>&1 | tee "${log_path}"
    else
      "${PYTHON_BIN}" "${ROOT_DIR}/scripts/npiv_syn/train_npiv_syn.py" \
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

# ── Aggregation ────────────────────────────────────────────────────
SUM_ARGS=(
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --out-fig "${OUT_FIG}"
  --design-seed "${DESIGN_SEED}"
  --data-seed "${DATA_SEED}"
)

if [[ -n "${SUM_TREATMENT_TYPES_CSV}" ]]; then
  IFS=',' read -r -a SUM_TTS <<< "${SUM_TREATMENT_TYPES_CSV}"
  SUM_ARGS+=(--treatment-types "${SUM_TTS[@]}")
else
  SUM_ARGS+=(--treatment-types "${TREATMENT_TYPES[@]}")
fi

if [[ -n "${SUM_SURFACES_CSV}" ]]; then
  IFS=',' read -r -a SUM_SURFS <<< "${SUM_SURFACES_CSV}"
  SUM_ARGS+=(--surfaces "${SUM_SURFS[@]}")
elif [[ -n "${EXPLICIT_SURFACE}" ]]; then
  SUM_ARGS+=(--surfaces "${EXPLICIT_SURFACE}")
fi

SUM_ARGS+=("${RUN_ARGS[@]}")

echo "=== test_npiv_syn -> ${OUT_FIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/npiv_syn/test_npiv_syn.py" "${SUM_ARGS[@]}"

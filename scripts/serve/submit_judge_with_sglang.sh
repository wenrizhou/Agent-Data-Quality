#!/bin/bash
#SBATCH --job-name=judge-sglang
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --account=polyullm
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/serve/submit_judge_with_sglang.sh" ]]; then
  PROJECT_DIR="$(cd -- "${SLURM_SUBMIT_DIR}" &>/dev/null && pwd)"
  SCRIPT_DIR="${PROJECT_DIR}/scripts/serve"
  SCRIPT_PATH="${SCRIPT_DIR}/submit_judge_with_sglang.sh"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
  PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
  SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
fi

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/serve/submit_judge_with_sglang.sh [options] [-- extra_run_judge_args...]

Starts a SGLang server in the Slurm job, waits until it is ready, runs
judge/run_judge.py against the local OpenAI-compatible API, then stops SGLang.

Options:
  --model-config PATH     SGLang model preset. Default: scripts/serve/models/qwen3-32b.sh
  --judge-config PATH     Judge config. Default: judge/configs/judge_api_all_metrics.yaml
  --input PATH            Override judge input path. Repeat for multiple paths.
  --max-samples N         Pass through to judge/run_judge.py.
  --balanced-sample       Pass through to judge/run_judge.py.
  --seed N                Pass through to judge/run_judge.py.
  --port PORT             SGLang HTTP port. Default: 31877
  --model PATH            Override MODEL from the SGLang preset.
  --model-name NAME       Override MODEL_NAME from the SGLang preset and use it for judge MODEL.
  --judge-model NAME      Override only the OpenAI-compatible API model name.
  --tp N                  Override tensor parallel size.
  --dp N                  Override data parallel size.
  --api-key KEY           API key passed to judge. Default: EMPTY
  --sglang-arg ARG        Extra SGLang launch argument. Repeat for multiple args.
  -h, --help              Show this help.

Container environment overrides:
  CONTAINER_IMAGE         Default: /lustre/projects/polyullm/container/slimerl+slime+latest+1202a.sqsh
  CONTAINER_MOUNTS        Default: /lustre/projects/polyullm:/lustre/projects/polyullm,/work/projects/polyullm:/work/projects/polyullm
  CONTAINER_NAME          Default: judge-sglang-${SLURM_JOB_ID}
  SKIP_SRUN_CONTAINER=1   Run directly instead of launching an inner srun container step.

Example:
  sbatch scripts/serve/submit_judge_with_sglang.sh \
    --input /work/projects/polyullm/shihao/agent/data/10_canonical/
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

run_inside_container_if_needed() {
  if [[ "${RUN_INSIDE_CONTAINER:-0}" == "1" || "${SKIP_SRUN_CONTAINER:-0}" == "1" ]]; then
    return
  fi
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "[job] SLURM_JOB_ID is not set; running in the current environment."
    return
  fi

  mkdir -p "${PROJECT_DIR}/logs/serve" "${PROJECT_DIR}/logs/judge"

  CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/projects/polyullm/container/slimerl+slime+latest+1202a.sqsh}"
  CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre/projects/polyullm:/lustre/projects/polyullm,/work/projects/polyullm:/work/projects/polyullm}"
  CONTAINER_NAME="${CONTAINER_NAME:-judge-sglang-${SLURM_JOB_ID}}"
  CONTAINER_ENV="${CONTAINER_ENV:-NVIDIA_DRIVER_CAPABILITIES,NVIDIA_VISIBLE_DEVICES,LOG_DIR,CACHE_DIR,HF_TOKEN,HUGGING_FACE_HUB_TOKEN}"

  export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
  export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"

  echo "[job] Launching container step on Slurm allocation"
  echo "[job] project=${PROJECT_DIR}"
  echo "[job] container=${CONTAINER_IMAGE}"

  INNER_CMD=(env RUN_INSIDE_CONTAINER=1 bash "${SCRIPT_PATH}" "$@")
  printf -v INNER_CMD_STR "%q " "${INNER_CMD[@]}"

  srun --nodes=1 --ntasks=1 \
    --container-name="${CONTAINER_NAME}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-workdir="${PROJECT_DIR}" \
    --container-writable \
    --container-remap-root \
    --container-env="${CONTAINER_ENV}" \
    bash -lc "${INNER_CMD_STR}"
  exit $?
}

resolve_file() {
  local raw="$1"
  local label="$2"

  if [[ "${raw}" = /* && -f "${raw}" ]]; then
    printf '%s\n' "${raw}"
  elif [[ -f "${PROJECT_DIR}/${raw}" ]]; then
    printf '%s\n' "${PROJECT_DIR}/${raw}"
  elif [[ -f "${raw}" ]]; then
    local dir
    dir="$(cd -- "$(dirname -- "${raw}")" && pwd)"
    printf '%s/%s\n' "${dir}" "$(basename -- "${raw}")"
  else
    echo "[job] ${label} not found: ${raw}" >&2
    exit 1
  fi
}

infer_model_name() (
  local config_file="$1"
  MODEL_NAME=""
  # shellcheck source=/dev/null
  source "${config_file}"
  printf '%s\n' "${MODEL_NAME:-}"
)

run_inside_container_if_needed "$@"

MODEL_CONFIG="${MODEL_CONFIG:-scripts/serve/models/qwen3-32b.sh}"
JUDGE_CONFIG="${JUDGE_CONFIG:-judge/configs/judge_api_all_metrics.yaml}"
PORT="${PORT:-31877}"
MODEL_OVERRIDE=""
MODEL_NAME_OVERRIDE=""
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
TP_OVERRIDE=""
DP_OVERRIDE=""
INPUT_PATHS=()
JUDGE_ARGS=()
SGLANG_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-config)
      MODEL_CONFIG="$2"
      shift 2
      ;;
    --judge-config)
      JUDGE_CONFIG="$2"
      shift 2
      ;;
    --input)
      INPUT_PATHS+=("$2")
      shift 2
      ;;
    --max-samples)
      JUDGE_ARGS+=(--max-samples "$2")
      shift 2
      ;;
    --balanced-sample)
      JUDGE_ARGS+=(--balanced-sample)
      shift
      ;;
    --seed)
      JUDGE_ARGS+=(--seed "$2")
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --model)
      MODEL_OVERRIDE="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME_OVERRIDE="$2"
      shift 2
      ;;
    --judge-model)
      JUDGE_MODEL="$2"
      shift 2
      ;;
    --tp)
      TP_OVERRIDE="$2"
      shift 2
      ;;
    --dp)
      DP_OVERRIDE="$2"
      shift 2
      ;;
    --api-key)
      JUDGE_API_KEY="$2"
      shift 2
      ;;
    --sglang-arg)
      SGLANG_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      JUDGE_ARGS+=("$@")
      break
      ;;
    *)
      echo "[job] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

MODEL_CONFIG="$(resolve_file "${MODEL_CONFIG}" "model config")"
JUDGE_CONFIG="$(resolve_file "${JUDGE_CONFIG}" "judge config")"
SERVE_SCRIPT="${SCRIPT_DIR}/serve_sglang.sh"
STOP_SCRIPT="${SCRIPT_DIR}/stop_sglang.sh"

LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/serve}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/cache/serve}"
mkdir -p "${LOG_DIR}" "${CACHE_DIR}" "${PROJECT_DIR}/logs/judge"
export LOG_DIR CACHE_DIR

PID_FILE="${PID_FILE:-${LOG_DIR}/judge_sglang_${PORT}_${SLURM_JOB_ID:-manual}.pid}"

if [[ -z "${JUDGE_MODEL}" ]]; then
  if [[ -n "${MODEL_NAME_OVERRIDE}" ]]; then
    JUDGE_MODEL="${MODEL_NAME_OVERRIDE}"
  else
    JUDGE_MODEL="$(infer_model_name "${MODEL_CONFIG}")"
  fi
fi
if [[ -z "${JUDGE_MODEL}" ]]; then
  echo "[job] Could not infer judge model name; pass --judge-model or --model-name." >&2
  exit 1
fi

cleanup() {
  local status=$?
  echo "[job] Cleaning up SGLang"
  bash "${STOP_SCRIPT}" --pid-file "${PID_FILE}" --timeout 30 --force || true
  exit "${status}"
}
trap cleanup EXIT

SERVE_CMD=(
  bash "${SERVE_SCRIPT}" "${MODEL_CONFIG}"
  --port "${PORT}"
  --pid-file "${PID_FILE}"
  --background
  --wait
)
if [[ -n "${MODEL_OVERRIDE}" ]]; then
  SERVE_CMD+=(--model "${MODEL_OVERRIDE}")
fi
if [[ -n "${MODEL_NAME_OVERRIDE}" ]]; then
  SERVE_CMD+=(--model-name "${MODEL_NAME_OVERRIDE}")
fi
if [[ -n "${TP_OVERRIDE}" ]]; then
  SERVE_CMD+=(--tp "${TP_OVERRIDE}")
fi
if [[ -n "${DP_OVERRIDE}" ]]; then
  SERVE_CMD+=(--dp "${DP_OVERRIDE}")
fi
if [[ "${#SGLANG_EXTRA_ARGS[@]}" -gt 0 ]]; then
  SERVE_CMD+=(-- "${SGLANG_EXTRA_ARGS[@]}")
fi

echo "[job] Starting SGLang"
echo "[job] model_config=${MODEL_CONFIG}"
echo "[job] port=${PORT}"
"${SERVE_CMD[@]}"

export MODEL="${JUDGE_MODEL}"
export BASE_URL="http://127.0.0.1:${PORT}/v1"
export API_KEY="${JUDGE_API_KEY}"

JUDGE_CMD=(python "${PROJECT_DIR}/judge/run_judge.py" --config "${JUDGE_CONFIG}")
if [[ "${#INPUT_PATHS[@]}" -gt 0 ]]; then
  JUDGE_CMD+=(--input "${INPUT_PATHS[@]}")
fi
if [[ "${#JUDGE_ARGS[@]}" -gt 0 ]]; then
  JUDGE_CMD+=("${JUDGE_ARGS[@]}")
fi

echo "[job] Running judge"
echo "[job] MODEL=${MODEL}"
echo "[job] BASE_URL=${BASE_URL}"
echo "[job] judge_config=${JUDGE_CONFIG}"
"${JUDGE_CMD[@]}"

echo "[job] Judge finished"

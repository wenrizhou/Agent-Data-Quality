#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash data_quality/scripts/serve/serve_sglang.sh <model_config.sh> [options] [-- extra_sglang_args...]

Options:
  --host HOST             Override HOST from config.
  --port PORT             Override PORT from config.
  --model MODEL           Override MODEL from config.
  --model-name NAME       Override MODEL_NAME from config.
  --tp N                  Override tensor parallel size.
  --dp N                  Override data parallel size.
  --background            Start server in background and write pid file.
  --wait                  Wait until /v1/models is ready after starting.
  --pid-file PATH         PID file path for --background.
  -h, --help              Show this help.

Examples:
  bash data_quality/scripts/serve/serve_sglang.sh \
    data_quality/scripts/serve/models/qwen3-4b-judge.sh

  bash data_quality/scripts/serve/serve_sglang.sh \
    data_quality/scripts/serve/models/qwen3-4b-judge.sh --background --wait
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

CONFIG_INPUT="$1"
shift

if [[ "${CONFIG_INPUT}" = /* ]]; then
  CONFIG_FILE="${CONFIG_INPUT}"
elif [[ -f "${CONFIG_INPUT}" ]]; then
  CONFIG_FILE="$(cd -- "$(dirname -- "${CONFIG_INPUT}")" && pwd)/$(basename -- "${CONFIG_INPUT}")"
elif [[ -f "${SCRIPT_DIR}/${CONFIG_INPUT}" ]]; then
  CONFIG_FILE="${SCRIPT_DIR}/${CONFIG_INPUT}"
elif [[ -f "${SCRIPT_DIR}/models/${CONFIG_INPUT}" ]]; then
  CONFIG_FILE="${SCRIPT_DIR}/models/${CONFIG_INPUT}"
else
  echo "[serve] Config file not found: ${CONFIG_INPUT}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

BACKGROUND=false
WAIT=false
PID_FILE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --tp)
      TP="$2"
      shift 2
      ;;
    --dp)
      DP="$2"
      shift 2
      ;;
    --background)
      BACKGROUND=true
      shift
      ;;
    --wait)
      WAIT=true
      shift
      ;;
    --pid-file)
      PID_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

: "${MODEL:?MODEL must be set by config or --model}"
: "${MODEL_NAME:?MODEL_NAME must be set by config or --model-name}"
: "${HOST:=0.0.0.0}"
: "${PORT:=31877}"
: "${TP:=1}"
: "${DP:=1}"
: "${WARMUPS:=3}"
: "${MAX_RUNNING_REQUESTS:=16}"
: "${CHUNKED_PREFILL_SIZE:=2048}"
: "${LOG_DIR:=${PROJECT_DIR}/logs/serve}"
: "${CACHE_DIR:=${PROJECT_DIR}/cache/serve}"

mkdir -p "${LOG_DIR}"
mkdir -p \
  "${CACHE_DIR}/xdg" \
  "${CACHE_DIR}/hf" \
  "${CACHE_DIR}/transformers" \
  "${CACHE_DIR}/flashinfer" \
  "${CACHE_DIR}/triton" \
  "${CACHE_DIR}/torchinductor" \
  "${CACHE_DIR}/cuda" \
  "${CACHE_DIR}/numba" \
  "${CACHE_DIR}/home" \
  "${CACHE_DIR}/sglang"

if [[ -z "${HOME:-}" || ! -w "${HOME}" ]]; then
  export HOME="${CACHE_DIR}/home"
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_DIR}/xdg}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_DIR}/transformers}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${CACHE_DIR}/flashinfer}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_DIR}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_DIR}/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${CACHE_DIR}/cuda}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${CACHE_DIR}/numba}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-${CACHE_DIR}/sglang}"

if [[ "${MODEL}" != /* && ! "${MODEL}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo "[serve] MODEL should be an absolute path or a HF repo id: ${MODEL}" >&2
  exit 1
fi

LAUNCH_ARGS=(
  python3 -m sglang.launch_server
  --model "${MODEL}"
  --served-model-name "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
  --tensor-parallel-size "${TP}"
  --data-parallel-size "${DP}"
  --warmups "${WARMUPS}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
)

if [[ -n "${TOOL_CALL_PARSER:-}" ]]; then
  LAUNCH_ARGS+=(--tool-call-parser "${TOOL_CALL_PARSER}")
fi
if [[ -n "${CONTEXT_LENGTH:-}" ]]; then
  LAUNCH_ARGS+=(--context-length "${CONTEXT_LENGTH}")
fi
if [[ -n "${MEM_FRACTION_STATIC:-}" ]]; then
  LAUNCH_ARGS+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
fi
if [[ -n "${LOG_LEVEL:-}" ]]; then
  LAUNCH_ARGS+=(--log-level "${LOG_LEVEL}")
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  LAUNCH_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "[serve] Config: ${CONFIG_FILE}"
echo "[serve] Model : ${MODEL}"
echo "[serve] Name  : ${MODEL_NAME}"
echo "[serve] URL   : http://${HOST}:${PORT}"
echo "[serve] Logs  : ${LOG_DIR}"
echo "[serve] Cache : ${CACHE_DIR}"
echo "[serve] Args  : ${LAUNCH_ARGS[*]}"

if [[ "${BACKGROUND}" == "true" ]]; then
  if [[ -z "${PID_FILE}" ]]; then
    PID_FILE="${LOG_DIR}/${MODEL_NAME}_${PORT}.pid"
  fi
  OUT_LOG="${LOG_DIR}/${MODEL_NAME}_${PORT}.out"
  ERR_LOG="${LOG_DIR}/${MODEL_NAME}_${PORT}.err"
  nohup "${LAUNCH_ARGS[@]}" >"${OUT_LOG}" 2>"${ERR_LOG}" &
  SERVER_PID=$!
  echo "${SERVER_PID}" >"${PID_FILE}"
  echo "[serve] Started in background: pid=${SERVER_PID}"
  echo "[serve] stdout: ${OUT_LOG}"
  echo "[serve] stderr: ${ERR_LOG}"
else
  "${LAUNCH_ARGS[@]}" &
  SERVER_PID=$!
  trap 'echo "[serve] stopping ${SERVER_PID}"; kill "${SERVER_PID}" 2>/dev/null || true' EXIT
fi

if [[ "${WAIT}" == "true" ]]; then
  WAIT_HOST="${HOST}"
  if [[ "${WAIT_HOST}" == "0.0.0.0" ]]; then
    WAIT_HOST="127.0.0.1"
  fi
  HOST="${WAIT_HOST}" PORT="${PORT}" bash "${SCRIPT_DIR}/wait_sglang.sh"
fi

if [[ "${BACKGROUND}" != "true" ]]; then
  wait "${SERVER_PID}"
fi

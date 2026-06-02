#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash data_quality/scripts/serve/stop_sglang.sh <model_config.sh> [options]
  bash data_quality/scripts/serve/stop_sglang.sh --pid-file PATH [options]

Options:
  --model-name NAME       Override MODEL_NAME from config.
  --port PORT             Override PORT from config.
  --pid-file PATH         PID file written by serve_sglang.sh --background.
  --log-dir PATH          Directory containing default PID files.
  --timeout SECONDS       Seconds to wait after SIGTERM. Default: 30.
  --force                 Send SIGKILL if the process does not stop in time.
  --dry-run               Print what would be stopped without sending signals.
  -h, --help              Show this help.

Examples:
  bash data_quality/scripts/serve/stop_sglang.sh \
    data_quality/scripts/serve/models/qwen3-4b-thinking.sh

  bash data_quality/scripts/serve/stop_sglang.sh \
    data_quality/scripts/serve/models/qwen3-4b-thinking.sh --force
EOF
}

CONFIG_INPUT=""
PID_FILE=""
TIMEOUT=30
FORCE=false
DRY_RUN=false

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

if [[ "${1:-}" != --* ]]; then
  CONFIG_INPUT="$1"
  shift
fi

if [[ -n "${CONFIG_INPUT}" ]]; then
  if [[ "${CONFIG_INPUT}" = /* ]]; then
    CONFIG_FILE="${CONFIG_INPUT}"
  elif [[ -f "${CONFIG_INPUT}" ]]; then
    CONFIG_FILE="$(cd -- "$(dirname -- "${CONFIG_INPUT}")" && pwd)/$(basename -- "${CONFIG_INPUT}")"
  elif [[ -f "${SCRIPT_DIR}/${CONFIG_INPUT}" ]]; then
    CONFIG_FILE="${SCRIPT_DIR}/${CONFIG_INPUT}"
  elif [[ -f "${SCRIPT_DIR}/models/${CONFIG_INPUT}" ]]; then
    CONFIG_FILE="${SCRIPT_DIR}/models/${CONFIG_INPUT}"
  else
    echo "[stop] Config file not found: ${CONFIG_INPUT}" >&2
    exit 1
  fi

  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --pid-file)
      PID_FILE="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[stop] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

: "${PORT:=31877}"
: "${LOG_DIR:=${PROJECT_DIR}/logs/serve}"

if [[ -z "${PID_FILE}" ]]; then
  : "${MODEL_NAME:?MODEL_NAME must be set by config, --model-name, or --pid-file}"
  PID_FILE="${LOG_DIR}/${MODEL_NAME}_${PORT}.pid"
fi

if [[ ! -f "${PID_FILE}" ]]; then
  echo "[stop] PID file not found: ${PID_FILE}"
  exit 0
fi

PID="$(tr -d '[:space:]' <"${PID_FILE}")"
if [[ ! "${PID}" =~ ^[0-9]+$ ]]; then
  echo "[stop] Invalid PID file content in ${PID_FILE}: ${PID}" >&2
  exit 1
fi

if ! kill -0 "${PID}" 2>/dev/null; then
  echo "[stop] Process is not running: pid=${PID}"
  rm -f "${PID_FILE}"
  echo "[stop] Removed stale PID file: ${PID_FILE}"
  exit 0
fi

CMD="$(ps -p "${PID}" -o args= 2>/dev/null || true)"
if [[ -n "${CMD}" ]]; then
  if [[ "${CMD}" != *"sglang.launch_server"* && "${CMD}" != *"sglang"* ]]; then
    echo "[stop] Refusing to stop pid=${PID}; command does not look like SGLang:" >&2
    echo "[stop] ${CMD}" >&2
    exit 1
  fi
fi

echo "[stop] PID file: ${PID_FILE}"
echo "[stop] pid=${PID}"
if [[ -n "${CMD}" ]]; then
  echo "[stop] cmd=${CMD}"
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[stop] dry-run: no signal sent"
  exit 0
fi

echo "[stop] Sending SIGTERM"
kill "${PID}" 2>/dev/null || true

for ((i = 0; i < TIMEOUT; i++)); do
  if ! kill -0 "${PID}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "[stop] Stopped and removed PID file"
    exit 0
  fi
  sleep 1
done

if [[ "${FORCE}" == "true" ]]; then
  echo "[stop] Still running after ${TIMEOUT}s; sending SIGKILL"
  kill -KILL "${PID}" 2>/dev/null || true
  rm -f "${PID_FILE}"
  echo "[stop] Removed PID file"
  exit 0
fi

echo "[stop] Still running after ${TIMEOUT}s: pid=${PID}" >&2
echo "[stop] Re-run with --force to send SIGKILL." >&2
exit 1

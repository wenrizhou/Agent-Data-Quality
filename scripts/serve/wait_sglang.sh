#!/bin/bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-31877}"
ATTEMPTS="${ATTEMPTS:-120}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"

if [[ $# -gt 0 ]]; then
  HOST="$1"
fi
if [[ $# -gt 1 ]]; then
  PORT="$2"
fi

URL="http://${HOST}:${PORT}/v1/models"

echo "[wait] Waiting for SGLang: ${URL}"
for ((i = 1; i <= ATTEMPTS; i++)); do
  if curl -fsS "${URL}" >/dev/null 2>&1; then
    echo "[wait] SGLang ready after ${i} attempt(s)"
    exit 0
  fi
  sleep "${SLEEP_SECONDS}"
done

echo "[wait] SGLang not ready after ${ATTEMPTS} attempt(s): ${URL}" >&2
exit 1

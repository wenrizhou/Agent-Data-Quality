#!/bin/bash
#SBATCH --job-name=sglang-serve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --account=polyullm
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/serve/submit_sglang.sh <model_config.sh> [serve_sglang options]

Example:
  ATTEMPTS=720 SLEEP_SECONDS=5 sbatch scripts/serve/submit_sglang.sh \
    scripts/serve/models/qwen3-4b-judge.sh --wait
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/serve/submit_sglang.sh" ]]; then
  WORKDIR="$(cd -- "${SLURM_SUBMIT_DIR}" &>/dev/null && pwd)"
  SCRIPT_DIR="${WORKDIR}/scripts/serve"
else
  WORKDIR="$(pwd)"
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
fi
mkdir -p "${WORKDIR}/logs/serve"

CONFIG_INPUT="$1"
shift

if [[ "${CONFIG_INPUT}" = /* ]]; then
  CONFIG_FILE="${CONFIG_INPUT}"
elif [[ -f "${WORKDIR}/${CONFIG_INPUT}" ]]; then
  CONFIG_FILE="${WORKDIR}/${CONFIG_INPUT}"
elif [[ -f "${SCRIPT_DIR}/${CONFIG_INPUT}" ]]; then
  CONFIG_FILE="${SCRIPT_DIR}/${CONFIG_INPUT}"
elif [[ -f "${SCRIPT_DIR}/models/${CONFIG_INPUT}" ]]; then
  CONFIG_FILE="${SCRIPT_DIR}/models/${CONFIG_INPUT}"
else
  echo "[submit] Config file not found: ${CONFIG_INPUT}" >&2
  exit 1
fi

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/projects/polyullm/container/slimerl+slime+latest+1202a.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre/projects/polyullm:/lustre/projects/polyullm,/work/projects/polyullm:/work/projects/polyullm}"
CONTAINER_NAME="${CONTAINER_NAME:-agent-judge-sglang-${SLURM_JOB_ID:-manual}}"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node="${nodes_array[0]}"
head_node_ip="$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address 2>/dev/null || hostname -I | awk '{print $1}')"

echo "[submit] workdir=${WORKDIR}"
echo "[submit] head_node=${head_node}"
echo "[submit] head_node_ip=${head_node_ip}"
echo "[submit] config=${CONFIG_FILE}"
echo "[submit] container=${CONTAINER_IMAGE}"

export NVIDIA_DRIVER_CAPABILITIES=all
export NVIDIA_VISIBLE_DEVICES=all
export ATTEMPTS="${ATTEMPTS:-120}"
export SLEEP_SECONDS="${SLEEP_SECONDS:-5}"

SERVE_SCRIPT="${SCRIPT_DIR}/serve_sglang.sh"
SERVE_CMD=(bash "${SERVE_SCRIPT}" "${CONFIG_FILE}" --host 0.0.0.0 "$@")
printf -v SERVE_CMD_STR "%q " "${SERVE_CMD[@]}"

srun --nodes=1 --ntasks=1 -w "${head_node}" \
  --container-name="${CONTAINER_NAME}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-image="${CONTAINER_IMAGE}" \
  --container-workdir="${WORKDIR}" \
  --container-writable \
  --container-remap-root \
  --container-env=NVIDIA_DRIVER_CAPABILITIES,NVIDIA_VISIBLE_DEVICES,ATTEMPTS,SLEEP_SECONDS \
  bash -lc "${SERVE_CMD_STR}"

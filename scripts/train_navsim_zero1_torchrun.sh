#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${PROJECT_ROOT}/data/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${PROJECT_ROOT}/runs}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${PROJECT_ROOT}/navsim}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${PROJECT_ROOT}/data/navsim}"
export PYTHONPATH="${PROJECT_ROOT}/src:${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

export NAVSIM_LOG_PATH="${OPENSCENE_DATA_ROOT}/navsim_logs/trainval"
export NAVSIM_SENSOR_BLOBS_PATH="${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval"
INFERRED_STORAGE_ROOT="$(cd "${OPENSCENE_DATA_ROOT}/.." && pwd -P)"
export NAVSIM_TEXT_EMBED_CACHE="${NAVSIM_TEXT_EMBED_CACHE:-${INFERRED_STORAGE_ROOT}/text_embeds_cache/navsim}"

for required_dir in \
  "${NUPLAN_MAPS_ROOT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim" \
  "${NAVSIM_LOG_PATH}" \
  "${NAVSIM_SENSOR_BLOBS_PATH}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Error: required directory does not exist: ${required_dir}" >&2
    exit 1
  fi
done

EXTRA_ARGS=("$@")

first_non_empty() {
  for value in "$@"; do
    if [[ -n "${value}" ]]; then
      echo "${value}"
      return 0
    fi
  done
  return 1
}

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

NPROC_PER_NODE="$(
  first_non_empty \
    "${NPROC_PER_NODE:-}" \
    "${GPU_NUM:-}" \
    "1"
)"
MAIN_PROCESS_IP="$(
  first_non_empty \
    "${MASTER_ADDR:-}" \
    "127.0.0.1"
)"
MAIN_PROCESS_PORT="$(
  first_non_empty \
    "${MASTER_PORT:-}" \
    "29500"
)"
MACHINE_RANK="$(
  first_non_empty \
    "${NODE_RANK:-}" \
    "${RANK:-}" \
    "0"
)"
NUM_MACHINES_RAW="$(
  first_non_empty \
    "${NNODES:-}" \
    "${WORLD_SIZE:-}" \
    "1"
)"

if ! is_integer "${NPROC_PER_NODE}" || ! is_integer "${MAIN_PROCESS_PORT}" || ! is_integer "${MACHINE_RANK}" || ! is_integer "${NUM_MACHINES_RAW}"; then
  echo "Error: NPROC_PER_NODE (${NPROC_PER_NODE}), MAIN_PROCESS_PORT (${MAIN_PROCESS_PORT}), MACHINE_RANK (${MACHINE_RANK}), and NUM_MACHINES_RAW (${NUM_MACHINES_RAW}) must be integers." >&2
  exit 1
fi

NUM_MACHINES="${NUM_MACHINES_RAW}"
if [[ -z "${NNODES:-}" ]] && [[ -n "${WORLD_SIZE:-}" ]] && (( NPROC_PER_NODE > 0 )); then
  if (( WORLD_SIZE % NPROC_PER_NODE == 0 )); then
    NUM_MACHINES="$(( WORLD_SIZE / NPROC_PER_NODE ))"
  else
    NUM_MACHINES="${WORLD_SIZE}"
  fi
fi

if ! is_integer "${NUM_MACHINES}"; then
  echo "Error: NUM_MACHINES (${NUM_MACHINES}) must be an integer." >&2
  exit 1
fi

if (( NUM_MACHINES > 1 )) && [[ -z "${MAIN_PROCESS_IP}" ]]; then
  echo "Error: MASTER_ADDR is empty in multi-machine mode." >&2
  exit 1
fi

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done

# LightWAM-AD training is local-model-only by default.  DiffSynth resolves model files as
#   ${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.1-T2V-1.3B/<file>
# Fail before torchrun starts four ranks instead of allowing every rank to race through a
# ModelScope snapshot lock.  Online download remains an explicit opt-in for other setups.
if [[ "${TASK_BASENAME}" == lightwam_ad_* ]]; then
  LIGHTWAM_MODEL_ID="Wan-AI/Wan2.1-T2V-1.3B"
  LIGHTWAM_ALLOW_DOWNLOAD="${LIGHTWAM_AD_ALLOW_MODEL_DOWNLOAD:-false}"

  if [[ -z "${DIFFSYNTH_MODEL_BASE_PATH:-}" ]]; then
    if [[ -n "${WAN_PRETRAINED_ROOT:-}" ]] && \
       [[ -d "${WAN_PRETRAINED_ROOT}" ]] && \
       compgen -G "${WAN_PRETRAINED_ROOT}/diffusion_pytorch_model*.safetensors" >/dev/null; then
      # WAN_PRETRAINED_ROOT points at the concrete model directory.
      DIFFSYNTH_MODEL_BASE_PATH="$(cd "${WAN_PRETRAINED_ROOT}/../.." && pwd -P)"
    else
      INFERRED_MODEL_BASE="$(cd "${OPENSCENE_DATA_ROOT}/.." && pwd -P)/models"
      if [[ -d "${INFERRED_MODEL_BASE}/${LIGHTWAM_MODEL_ID}" ]]; then
        DIFFSYNTH_MODEL_BASE_PATH="${INFERRED_MODEL_BASE}"
      fi
    fi
  fi

  if [[ "${LIGHTWAM_ALLOW_DOWNLOAD,,}" == "true" ]]; then
    export DIFFSYNTH_SKIP_DOWNLOAD=false
    echo "[model] LightWAM online download explicitly enabled."
  else
    if [[ -z "${DIFFSYNTH_MODEL_BASE_PATH:-}" ]]; then
      echo "Error: cannot resolve local Wan model base." >&2
      echo "Set DIFFSYNTH_MODEL_BASE_PATH to the directory containing ${LIGHTWAM_MODEL_ID}." >&2
      exit 1
    fi
    LIGHTWAM_MODEL_DIR="${DIFFSYNTH_MODEL_BASE_PATH}/${LIGHTWAM_MODEL_ID}"
    if ! compgen -G "${LIGHTWAM_MODEL_DIR}/diffusion_pytorch_model*.safetensors" >/dev/null; then
      echo "Error: missing ${LIGHTWAM_MODEL_DIR}/diffusion_pytorch_model*.safetensors" >&2
      exit 1
    fi
    if [[ ! -f "${LIGHTWAM_MODEL_DIR}/Wan2.1_VAE.pth" ]]; then
      echo "Error: missing ${LIGHTWAM_MODEL_DIR}/Wan2.1_VAE.pth" >&2
      exit 1
    fi
    if [[ ! -d "${NAVSIM_TEXT_EMBED_CACHE}" ]] || \
       ! compgen -G "${NAVSIM_TEXT_EMBED_CACHE}/*.pt" >/dev/null; then
      echo "Error: no cached text embedding found under ${NAVSIM_TEXT_EMBED_CACHE}" >&2
      echo "Run scripts/precomput_text_embed.sh before training." >&2
      exit 1
    fi
    export DIFFSYNTH_MODEL_BASE_PATH
    export DIFFSYNTH_SKIP_DOWNLOAD=true
    echo "[model] local_only=true base=${DIFFSYNTH_MODEL_BASE_PATH} model_dir=${LIGHTWAM_MODEL_DIR}"
    echo "[model] text_embed_cache=${NAVSIM_TEXT_EMBED_CACHE}"
  fi
fi

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"

    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT
    export RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"

    RUN_ID="$(
      python - <<'PY'
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_machines,
    is_master=(machine_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
run_id = store.get(key).decode("utf-8")
print(run_id)
PY
    )"

    echo "[run_id_sync] mode=tcpstore host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} timeout_s=${RUN_ID_SYNC_TIMEOUT} run_id=${RUN_ID}"
  fi
fi

# Make Accelerator pick up DeepSpeed under torchrun without relying on accelerate launch.
export ACCELERATE_USE_DEEPSPEED=true
# Honor a caller-provided DeepSpeed config (e.g. ZeRO-2 + optimizer CPU offload for memory-heavy
# runs like IDM); default to ZeRO-1 when unset.
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${ACCELERATE_DEEPSPEED_CONFIG_FILE:-${PROJECT_ROOT}/scripts/ds_configs/ds_zero1_config.json}"

echo "[launch] launcher=torchrun main_process_ip=${MAIN_PROCESS_IP} main_process_port=${MAIN_PROCESS_PORT} nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} world_size=$(( NUM_MACHINES * NPROC_PER_NODE )) run_id=${RUN_ID}"
echo "[navsim] task=${TASK_BASENAME}"
echo "[navsim] OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
echo "[navsim] NAVSIM_LOG_PATH=${NAVSIM_LOG_PATH}"
echo "[navsim] NAVSIM_SENSOR_BLOBS_PATH=${NAVSIM_SENSOR_BLOBS_PATH}"
echo "[dist-env] MASTER_ADDR=${MAIN_PROCESS_IP} MASTER_PORT=${MAIN_PROCESS_PORT} NNODES=${NUM_MACHINES} NODE_RANK=${MACHINE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE} WORLD_SIZE=${WORLD_SIZE:-unset}"

if [[ "${LIGHTWAM_AD_PREFLIGHT_ONLY:-false}" == "true" ]]; then
  echo "[preflight] PASS: data roots and local LightWAM-AD weights are available; torchrun was not started."
  exit 0
fi

cd "${PROJECT_ROOT}"
torchrun \
  --nnodes "${NUM_MACHINES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MAIN_PROCESS_PORT}" \
  "./scripts/train.py" \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${PROJECT_ROOT}/data/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${PROJECT_ROOT}/runs}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${PROJECT_ROOT}/navsim}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${PROJECT_ROOT}/data/navsim}"
export NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/trainval}"
export NAVSIM_SENSOR_BLOBS_PATH="${NAVSIM_SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval}"
export NAVSIM_TEXT_EMBED_CACHE="${NAVSIM_TEXT_EMBED_CACHE:-${PROJECT_ROOT}/data/text_embeds_cache/navsim}"
export PYTHONPATH="${PROJECT_ROOT}/src:${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
TASK="${TASK:-lightwam_ad_navsim_front_384x672}"

if [[ "${TASK}" == lightwam_ad_* ]]; then
  if [[ -z "${DIFFSYNTH_MODEL_BASE_PATH:-}" ]]; then
    INFERRED_MODEL_BASE="$(cd "${OPENSCENE_DATA_ROOT}/.." && pwd -P)/models"
    if [[ -d "${INFERRED_MODEL_BASE}/Wan-AI/Wan2.1-T2V-1.3B" ]]; then
      export DIFFSYNTH_MODEL_BASE_PATH="${INFERRED_MODEL_BASE}"
    else
      echo "Error: cannot resolve local Wan model base." >&2
      echo "Set DIFFSYNTH_MODEL_BASE_PATH to the directory containing Wan-AI/Wan2.1-T2V-1.3B." >&2
      exit 1
    fi
  fi
  MODEL_DIR="${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.1-T2V-1.3B"
  if [[ ! -f "${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth" ]]; then
    echo "Error: missing ${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth" >&2
    exit 1
  fi
  if [[ ! -d "${MODEL_DIR}/google/umt5-xxl" ]]; then
    echo "Error: missing tokenizer directory ${MODEL_DIR}/google/umt5-xxl" >&2
    exit 1
  fi
  export DIFFSYNTH_SKIP_DOWNLOAD=true
  echo "[model] text-cache local_only=true model_dir=${MODEL_DIR}"
fi

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

cd "${PROJECT_ROOT}"
python scripts/precompute_navsim_text_embeds.py \
  "task=${TASK}" \
  "$@"

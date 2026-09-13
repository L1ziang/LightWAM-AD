#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
: "${CKPT:?Set CKPT to the LightWAM-AD weights checkpoint.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${PROJECT_ROOT}/data/navsim}"
export NAVSIM_DEVKIT_ROOT="${PROJECT_ROOT}/navsim_v2"
export NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}"
export NAVSIM_SENSOR_BLOBS_PATH="${NAVSIM_SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${PROJECT_ROOT}/benchmark_results}"
export PYTHONPATH="${PROJECT_ROOT}/navsim_v2:${PROJECT_ROOT}/src:${SCRIPT_DIR}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BENCH_OUT="${BENCH_OUT:-${PROJECT_ROOT}/benchmark_results/navsim_$(date +%Y%m%d_%H%M%S)}"
exec "${PYTHON_BIN}" -u experiments/navsim/benchmark_lightwam.py \
  "ckpt=${CKPT}" \
  "BENCHMARK.output_dir=${BENCH_OUT}" \
  "BENCHMARK.cpu_threads=${OMP_NUM_THREADS}" \
  "$@"

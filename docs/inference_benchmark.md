# Inference latency benchmark

Measures a trained checkpoint on real NAVSIM-v2 front images using **one CUDA
GPU, batch 1, BF16**. Follow [setup & usage](getting_started.md) to prepare the
model, text cache and test data; metric caches and PDM scoring are unnecessary.
The recorded results are in the [e100 H200 report](results/lightwam_ad_e100_h200.md).

## What is timed

| Measurement | Start → end |
| --- | --- |
| Policy | Preprocessed CPU image, ego state and cached text → normalized CPU float32 action |
| E2E | In-memory RGB, prepared ego state and cached text → physical CPU trajectory |

Both include input transfers, context preparation, VAE encoding, backbone,
trajectory head and output transfer. E2E also includes image resize/normalization
and trajectory denormalization. Scene/JPEG/text-cache loading, model loading,
PDM scoring and file output are excluded. Images are preloaded as raw RGB, not
cached visual features. This is not sensor-to-actuator latency.

Default protocol from [benchmark_navsim.yaml](../configs/benchmark_navsim.yaml):

- 200 distinct navtest scenes selected reproducibly with seed 42.
- 3 repeats, 30 warmup requests per variant; variant order alternates by repeat.
- Synchronized, uninstrumented wall time under `eval` and `no_grad`; no
  `torch.compile`, LoRA merge or future generation.
- Separate CUDA-event profiling on 30 scenes. Regions measure device-timeline
  elapsed time including launch gaps, not pure kernel sums. Resamplers and
  decoder are nested inside the full head region; do not sum them twice.
- Peak allocated/reserved memory measured after warmup, without per-request
  cache clearing. PyTorch allocated memory differs from total process memory.

`baseline` executes all 30 backbone blocks. `early_exit` sets
`infer_action(skip_unused_tail=True)` and executes 25, stopping after the last
zero-indexed tap `[8,16,24]`. It retains the same loaded weights, including the
unused VAE decoder and tail. Normal inference defaults to `baseline`.

Before comparison, all selected scenes must pass trajectory parity with a default
normalized tolerance of **0**. Mismatches or nonfinite outputs abort comparative
timing and save diagnostics. Sampled parity is not a full NAVSIM reevaluation.
The parameter report distinguishes the complete loaded model from parameters
trainable under the model configuration; neither count measures FLOPs.

## Run

On an idle GPU, with the inference environment already activated:

```bash
cd /path/to/LightWAM-AD
export CKPT=/path/to/checkpoints/weights/step_131700.pt
export OPENSCENE_DATA_ROOT=/path/to/assets/navsim
export NAVSIM_TEXT_EMBED_CACHE=/path/to/assets/text_embeds_cache/navsim
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/assets/models
export DIFFSYNTH_SKIP_DOWNLOAD=true
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If the test logs, sensor blobs or maps use a different layout, explicitly set
`NAVSIM_LOG_PATH`, `NAVSIM_SENSOR_BLOBS_PATH` and `NUPLAN_MAPS_ROOT`.
`PYTHON_BIN` optionally selects a specific Python executable.

Optional smoke run:

```bash
export BENCH_OUT="$PWD/benchmark_results/smoke_$(date +%Y%m%d_%H%M%S)"
bash experiments/navsim/run_benchmark_lightwam.sh \
  BENCHMARK.num_samples=4 BENCHMARK.warmup=2 \
  BENCHMARK.repeats=1 BENCHMARK.profile_samples=2
```

Full measurement:

```bash
export BENCH_OUT="$PWD/benchmark_results/e100_gpu_$(date +%Y%m%d_%H%M%S)"
bash experiments/navsim/run_benchmark_lightwam.sh
cat "$BENCH_OUT/summary.md"
```

Use `'BENCHMARK.variants=[baseline]'` for baseline only. Increase
`BENCHMARK.num_samples` for broader sampling; keep `profile_samples <= num_samples`.
Append `--cfg job --resolve` to inspect configuration without loading the model.
Each run requires an empty output directory. Other jobs on the selected GPU
can distort timings; GPU snapshots are recorded without changing clock/power limits.

## Saved artifacts

| File | Contents |
| --- | --- |
| `summary.md`, `summary.json` | Tables, per-repeat statistics, environment, commit and checkpoint SHA256 |
| `latency_samples.csv` | Per-token/variant/repeat wall timings |
| `cuda_profile.csv` | Separate module-profiling measurements |
| `parity.json`, `parity.csv`, `parity_predictions.npz` | Comparison results and paired physical trajectories; comparison runs only |
| `tokens.json`, `config.yaml` | Exact inputs and resolved configuration |
| `status.json`, `benchmark.log` | Completion status and diagnostics |

Keep the full output directory with any published summary. `benchmark_results/`
is ignored by Git; curated reports live in `docs/results/`. Compare external
models on the same GPU, inputs, precision and timing boundary before claiming a
speedup. Validation tests are listed in [architecture](architecture.md#verification).

# Setup and usage

Run these commands in **Bash on a Linux CUDA machine**, from the repository root,
with the environment activated. LightWAM-AD uses
`task=lightwam_ad_navsim_front_384x672`. This repository contains the NAVSIM
training, trajectory prediction, scoring and latency benchmark paths for this model.

## 1. Install

```bash
conda create -n lightwam-ad python=3.10 -y
conda activate lightwam-ad
python -m pip install -r requirements.txt
python -m pip install -e navsim_v2 --no-deps
python -m pip install -e . --no-deps
```

`requirements.txt` records the server environment, including PyTorch 2.7.1,
torchvision 0.22.1 and DeepSpeed. Both vendored devkits use the import name
`navsim`: launch scripts select `navsim/` for training and `navsim_v2/` for v2
evaluation through `PYTHONPATH`.

## 2. Prepare assets

Obtain NAVSIM/OpenScene data and nuPlan maps using the
[NAVSIM instructions](https://github.com/autonomousvision/navsim), and pretrained
[Wan2.1-T2V-1.3B files](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).
NAVSIM-v2 scoring also needs its split-specific metric caches; navhard prediction
needs the supplied synthetic scenes and sensor blobs. Prepare these with the v2
devkit; v1 metric caches cannot substitute for v2 caches.

Example layout (large assets can be symlinked):

```text
/path/to/assets/
├── models/Wan-AI/Wan2.1-T2V-1.3B/
│   ├── diffusion_pytorch_model*.safetensors
│   ├── Wan2.1_VAE.pth
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── google/umt5-xxl/                 # tokenizer files
├── text_embeds_cache/navsim/           # generated below
├── navsim/
│   ├── maps/
│   ├── navsim_logs/{trainval,test}/
│   ├── sensor_blobs/{trainval,test}/
│   └── navhard_two_stage/
│       ├── sensor_blobs/
│       └── synthetic_scene_pickles/
├── metric_cache_navtest_v2/
└── metric_cache_navhard_two_stage/
```

Set paths once in the current shell:

```bash
export ASSET_ROOT=/path/to/assets
export OPENSCENE_DATA_ROOT="$ASSET_ROOT/navsim"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_TEXT_EMBED_CACHE="$ASSET_ROOT/text_embeds_cache/navsim"
export DIFFSYNTH_MODEL_BASE_PATH="$ASSET_ROOT/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export LIGHTWAM_AD_ALLOW_MODEL_DOWNLOAD=false
export TOKENIZERS_PARALLELISM=false
```

T5 is needed to generate the cache, then omitted from training/inference memory:

```bash
TASK=lightwam_ad_navsim_front_384x672 \
bash scripts/precomput_text_embed.sh prompt_collect_workers=1 overwrite=false
```

The default static prompt can share its cache between training and evaluation.
Changing the prompt or text encoder requires a matching cache. Model and data
assets are not distributed with this repository.

## 3. Train

The task file has small-run defaults (3 epochs, batch 1, accumulation 2).
Use these explicit overrides for the **reported 100-epoch recipe**:

```bash
export NPROC_PER_NODE=8
export RUN_ID="lightwam_ad_main64_e100_$(date +%Y%m%d_%H%M%S)"
TRAIN_ARGS=(
  task=lightwam_ad_navsim_front_384x672
  batch_size=8 gradient_accumulation_steps=1 num_workers=8
  num_epochs=100 max_steps=null
  learning_rate=1e-4 lr_scheduler_type=cosine weight_decay=0.01
  mixed_precision=bf16 seed=42
  data.validation_split.enabled=true data.validation_split.val_fraction=0.01
  eval_every=1000 eval_num_batches=null
  save_every=2634 save_final_checkpoint=false log_every=10 wandb.enabled=false
)
LIGHTWAM_AD_PREFLIGHT_ONLY=true \
  bash scripts/train_navsim_zero1_torchrun.sh "${TRAIN_ARGS[@]}"
bash scripts/train_navsim_zero1_torchrun.sh "${TRAIN_ARGS[@]}"
```

The preflight checks local asset paths without starting training. Effective batch
size is GPUs × per-GPU batch × accumulation = **64**. With the recorded 84,258
training samples, the run has 1,317 optimizer steps per epoch and ends at
**131,700**. Different data selections or batch settings change this step count.
If changing them, use `save_final_checkpoint=true` to save the final weights even
when the last step does not land on `save_every`.

Training selects the configured `navtrain` scene filter and `train_logs` list;
the recorded selection contains 85,109 usable clips. A seed-42 token-hash split
holds out 851 for monitoring, leaving 84,258 for optimization. This is a subset
selection, not a claim to use all official navtrain scenes. Temporally adjacent
clips can overlap; use the separate evaluation splits for reported scores.

Runs, TensorBoard events and checkpoints are under
`runs/lightwam_ad_navsim_front_384x672/$RUN_ID/`. To resume, append
`resume=/path/to/checkpoints/state/step_XXXXXX` using an existing Accelerate state
directory. A `checkpoints/weights/step_*.pt` file does not include optimizer state.
The reported run resumed from saved training state; the command above starts fresh.

## 4. Predict on NAVSIM-v2

Use a new output directory per checkpoint. Prediction writes one physical
`[8,3]` trajectory `.npy` per token; scoring is a separate CPU step.

```bash
export CKPT=/path/to/checkpoints/weights/step_131700.pt
export TASK=lightwam_ad_navsim_front_384x672
export NPROC_PER_NODE=4  # set to the number of available GPUs
export EXP_NAME="lightwam_ad_e100_$(date +%Y%m%d_%H%M%S)"
export NAVSIM_EXP_ROOT="$PWD/evaluate_results_v2"
export NAVSIM_LOG_PATH="$OPENSCENE_DATA_ROOT/navsim_logs/test"
export NAVSIM_SENSOR_BLOBS_PATH="$OPENSCENE_DATA_ROOT/sensor_blobs/test"
export NAVTEST_PRED_DIR="$NAVSIM_EXP_ROOT/navtest/$EXP_NAME/pred_actions"
export NAVHARD_PRED_DIR="$NAVSIM_EXP_ROOT/navhard/$EXP_NAME/pred_actions"
export SYNTHETIC_SENSOR_PATH="$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs"
export SYNTHETIC_SCENES_PATH="$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles"

PRED_DIR="$NAVTEST_PRED_DIR" \
  bash experiments/navsim/run_predict_navsim_v2.sh

PRED_DIR="$NAVHARD_PRED_DIR" \
NAVHARD_SYNTHETIC_SENSOR_PATH="$SYNTHETIC_SENSOR_PATH" \
NAVHARD_SYNTHETIC_SCENES_PATH="$SYNTHETIC_SCENES_PATH" \
  bash experiments/navsim/run_predict_navhard.sh
```

The navhard launcher covers both original and synthetic scenes. Keep the model
task, action normalization and resolution consistent with training. The published
score record uses baseline inference, without the optional unused-tail skip.

## 5. Score predictions

Use the **v2** caches prepared for these splits. For example, with 16 CPU worker
processes (adjust to available cores and memory):

```bash
export NAVTEST_CACHE="$ASSET_ROOT/metric_cache_navtest_v2"
export NAVHARD_CACHE="$ASSET_ROOT/metric_cache_navhard_two_stage"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SPLIT=both WORKER=sequential \
bash navsim_v2/scripts/evaluation/run_npy_trajectory_agent_pdm_score_evaluation.sh \
  worker._target_=nuplan.planning.utils.multithreading.worker_parallel.SingleMachineParallelExecutor \
  +worker.use_process_pool=true +worker.max_workers=16
```

Here `WORKER=sequential` selects the existing config, then the explicit overrides
replace it with nuPlan's [process-pool executor](https://github.com/motional/nuplan-devkit/blob/nuplan-devkit-v1.1/nuplan/planning/utils/multithreading/worker_parallel.py).
The wrapper runs reactive navtest scoring followed by navhard two-stage scoring.
For the recorded 80-core / 900 GB allocation, 72 workers were used.

CSV files are written under
`$NAVSIM_EXP_ROOT/${EXP_NAME}_navtest_v2/` and
`$NAVSIM_EXP_ROOT/${EXP_NAME}_navhard_v2/` in timestamped subdirectories.
Check valid-scene counts (12,146 navtest; 450 + 5,462 navhard), then use
`average_all_frames` for navtest and `extended_pdm_score_combined` for navhard.
Multiply by 100 for the README scale; do not average navhard's two stage scores.

## 6. Measure latency

Follow the [single-GPU benchmark guide](inference_benchmark.md). Metric caches and
CPU scoring workers are unnecessary for that benchmark. The
[experiment record](results/lightwam_ad_e100_h200.md) contains the existing results.

# Architecture and data contract

Implementation: [lightwam_ad.py](../src/simwam/models/wan22/lightwam_ad.py).
The `simwam` Python namespace is retained from the upstream codebase.
The reported model uses the [rank-64 model config](../configs/model/lightwam_ad_navsim.yaml)
and [front-camera task](../configs/task/lightwam_ad_navsim_front_384x672.yaml).

## Inputs and targets

| Tensor | Training shape | Meaning |
| --- | --- | --- |
| `video` | `[B,3,9,384,672]` | Current + 8 future front frames, normalized to `[-1,1]` |
| `action` | `[B,8,3]` | Normalized absolute `(x,y,heading)` in the current ego frame |
| `proprio` | `[B,8,8]` | Current velocity, acceleration and navigation command, repeated along the action axis |
| `context` | `[B,256,4096]` | Cached T5 embedding of the static driving prompt |
| VAE latents | `[B,16,3,48,84]` | One current latent frame + two future latent frames |

Images are sampled at `t = 0, 0.5, ..., 4.0 s`; action targets start at `0.5 s`.
The dataset enforces `future_action_horizon == (num_frames - 1) * frame_stride`.
It loads the front camera only. The current state and coordinate origin refer to
the same frame. The default prompt is static (`use_dynamic_prompt=false`), so
navigation information enters through ego state rather than changing the text.

Action normalization uses the fixed affine transforms in
[NavSimVideoDataset](../src/simwam/datasets/navsim/navsim_dataset.py), shared by
training and prediction. Dataset statistics files are not used by this path.
The trajectory head's ego normalization defaults to identity.

## Training

The frozen, causal Wan VAE compresses nine frames into three latent frames.
The current latent stays clean at timestep zero; noise is added to the future
latents for flow matching. A single Wan DiT pass processes the clip with
`first_frame_causal` self-attention:

```text
current queries -> current keys only
future queries  -> current and future keys
```

Zero-initialized residual adapters update the features at zero-indexed blocks
`[8,16,24]`. Current-frame features are captured after each adapter; adapted
features also continue through the backbone. The mask isolates action inputs
from future tokens while future-video supervision trains the shared weights.

Each tapped layer has eight learned resampler queries of width 512. The three
sets of slots remain separate, giving 24 visual tokens plus one ego-state token.
Eight waypoint queries attend to this memory through a two-layer Transformer
decoder. A linear projection emits the entire trajectory without an output clamp.
Ego state also enters the backbone through a projection to text-context width.

```text
loss = future-latent flow MSE + trajectory Smooth L1
```

Both loss weights are 1; Smooth L1 beta is 0.1. Video loss covers all channels and
spatial cells in the two future latent frames, excluding the current frame.
Padding masks exclude invalid targets. Spatial downsampling is **1** in the
reported run; factor 2 remains an optional, unreported ablation.

## Inference

The VAE encodes one current image. One backbone pass supplies the same three
feature taps to the trajectory head, which predicts eight poses at once.
There are no future latent tokens, generated future frames or iterative action
sampling steps. The unit tests compare the current representations from full-clip
and single-frame execution and check isolation from future-input gradients.

The optional `infer_action(skip_unused_tail=True)` stops after block 24, executing
25 of 30 blocks. Normal inference defaults to the full path; training always
uses all 30 blocks. The skip requires evaluation without backbone gradients.
See the [benchmark protocol](inference_benchmark.md) for parity checks.

## Parameters and checkpoints

The pretrained DiT is frozen except for its video output head. Rank-64,
alpha-128 LoRA is trained on self/cross-attention `q/k/v/o` and FFN `0/2` linear
projections in **all 30 blocks**. Adapters, trajectory head and ego projection
are also trained. The VAE is frozen; cached text avoids loading T5 during
training and inference.

| Loaded component | Parameters | Trainable |
| --- | ---: | ---: |
| Original Wan DiT, including video output head | 1,418,996,800 | 101,440 |
| Added LoRA | 87,490,560 | 87,490,560 |
| Three residual adapters | 2,373,888 | 2,373,888 |
| Trajectory head, including resamplers and state token | 20,552,707 | 20,552,707 |
| Ego-to-context projection | 36,864 | 36,864 |
| VAE encoder and decoder | 126,892,531 | 0 |
| **Total** | **1,656,343,350** | **110,555,459** |

The component breakdown follows the model configuration; both totals were
confirmed by the server benchmark. B/M denote decimal parameter counts. Loaded
parameters include the unused VAE decoder, video output head and backbone tail;
the optional early exit removes execution only. Counts are not FLOPs or peak
memory estimates.

Weights use format `lightwam_ad_v1`. With a frozen backbone, a weights file stores
LoRA, the video output head and the added modules, and therefore still requires
the original Wan DiT/VAE files. Checkpoint loading validates compatibility,
including tap identities. Resume training from the separate Accelerate state
directory to restore optimizer and scheduler state; `weights/step_*.pt` is for
model loading, not full training resumption.

The Wan2.1 loader uses the 16-channel, hidden-1536, 12-head, 30-block preset. It
checks shape-compatible keys and parameter elements against a 90% minimum
coverage before accepting pretrained weights.

## Verification

```bash
PYTHONPATH=src python -m pytest -q tests/test_lightwam_ad.py tests/test_inference_benchmark.py
```

Tests cover attention masks, feature parity, future-gradient isolation, latent
loss alignment, slot preservation, trainable parameter policy, PEFT checkpoint
round trips and unused-tail behavior. GPU latency requires the separate server
benchmark; CPU tests do not measure inference speed.

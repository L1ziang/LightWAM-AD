---
base_model: Wan-AI/Wan2.1-T2V-1.3B
base_model_relation: adapter
tags:
  - autonomous-driving
  - navsim
  - world-model
  - lora
---

# LightWAM-AD: epoch 100

A one-pass driving policy adapted from Wan2.1-T2V-1.3B with future-video
supervision and direct trajectory regression.
[Code and architecture](https://github.com/L1ziang/LightWAM-AD/tree/lightwam-ad).

## Checkpoint and dependencies

`step_131700.pt` uses the custom `lightwam_ad_v1` format. It contains LoRA,
residual adapters, the trajectory head, the trained video output head and the
ego-state projection. It is an adaptation checkpoint, not a standalone model
or a standard Transformers/PEFT `from_pretrained` package. Optimizer state is
not included.

The original **Wan DiT, VAE, T5 and tokenizer are not redistributed here**.
Download them from the official
[Wan2.1-T2V-1.3B repository](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).
DiT and VAE are needed for inference. T5 and its tokenizer prepare the static
text-embedding cache; once the matching cache exists, inference does not load T5.
NAVSIM data, maps and scoring caches must also be obtained separately.

Download this checkpoint:

```bash
hf download {{REPO_ID}} step_131700.pt SHA256SUMS --local-dir ./lightwam-ad-e100
```

Follow the [setup, cache preparation and evaluation guide](https://github.com/L1ziang/LightWAM-AD/blob/lightwam-ad/docs/getting_started.md),
using `task=lightwam_ad_navsim_front_384x672` and
`CKPT=/path/to/lightwam-ad-e100/step_131700.pt`.
The fixed action normalization is implemented in the codebase.

## Training and results

NAVSIM/OpenScene adaptation: 84,258 training and 851 validation clips;
100 epochs, effective batch 64, rank-64 LoRA and BF16. Training uses current
and future front images with expert trajectories; no SimScale data or RL.
Inference takes the current 384 × 672 front image, ego state and cached text,
and outputs eight ego-frame poses over four seconds.

| NAVSIM-v2 EPDMS (0–100) | Score |
| --- | ---: |
| navtest | 88.576 |
| navhard, combined | 33.281 |

On one H200, batch 1 and BF16: mean E2E latency is 71.570 ms, or 63.575 ms
with the optional unused-tail skip. E2E includes in-memory preprocessing and
transfers, but excludes disk loading and PDM scoring. Full-model parameters:
1,656,343,350 loaded and 110,555,459 trainable. These counts include the
separately obtained base model, not just this weights file.

Scores use the baseline inference path. The optional skip has exact output
parity on 200 sampled scenes, not a separate full NAVSIM evaluation.
[Full experiment record](https://github.com/L1ziang/LightWAM-AD/blob/lightwam-ad/docs/results/lightwam_ad_e100_h200.md).
This is a research checkpoint; the reported evaluations do not establish
real-vehicle deployment readiness.

## Attribution

We thank [SimWAM](https://github.com/H-EmbodVis/SimWAM) for the codebase and
training/evaluation infrastructure, and the Wan, Light-WAM, Fast-WAM and
NAVSIM teams. The codebase retains its MIT license and upstream notices;
the separately downloaded Wan assets retain their Apache-2.0 license.

# LightWAM-AD e100: NAVSIM-v2 scores and H200 inference

This record contains the reported server evaluation and benchmark summaries.
Raw timing samples, resolved run configuration, scene tokens, GPU snapshots and
score CSVs remain in the server output directories and are not bundled here.
See the [README](../../README.md#results-lightwam-ad-and-simwam) for the SimWAM
reference comparison and [usage guide](../getting_started.md) for reproduction.

## Checkpoint and setup

| Item | Value |
| --- | --- |
| Training configuration | Rank-64 LoRA, 8 GPUs × batch 8, gradient accumulation 1, 100 epochs |
| Weights | `step_131700.pt`, epoch 100 |
| Checkpoint SHA256 | `7b880759c6c4244cb9123cfd9320eed303dd9b6204979733fbfe4c96b95c52c5` |
| GPU | NVIDIA H200 |
| Batch / precision | 1 / `torch.bfloat16` |
| Main model recipe | Front camera, current frame, 384 × 672; rank-64 LoRA; taps `[8,16,24]`; 8 trajectory poses over 4 seconds |
| Compilation / LoRA merge | Neither used |

## Epoch 100: NAVSIM-v2 evaluation

These scores evaluate `step_131700.pt` from a single rank-64 training run, rather
than an average over seeds. They use the original full-backbone inference path;
the latency benchmark and score CSVs are separate server artifacts.

The reported successful-scene counts are **12,146 for navtest** and **5,912 for
navhard** (total across both stages).

| Split | Aggregation row | Score (0–1) | Reported score (0–100) |
| --- | --- | ---: | ---: |
| navtest | `average_all_frames` | 0.885763 | 88.576 |
| navhard | `extended_pdm_score_stage_one` | 0.775706 | 77.571 |
| navhard | `extended_pdm_score_stage_two` | 0.425914 | 42.591 |
| navhard | `extended_pdm_score_combined` | 0.332815 | 33.281 |

The combined score is the scorer's reported aggregation, not an
arithmetic average of the two stage scores. Displayed score precisions follow the
original report.

### Component metrics

Official summary rows, scaled to 0–100. S1 has 450 scenes; S2 has 5,462.
All scene scores were valid and finite, with no duplicate tokens. S2 values use
the official aggregation, which differs from an unweighted mean of CSV scene rows.

| Metric | navtest | navhard S1 | navhard S2 |
| --- | ---: | ---: | ---: |
| No at-fault collision (NC) | 98.172 | 97.111 | 81.010 |
| Drivable area compliance (DAC) | 96.773 | 88.444 | 73.720 |
| Driving direction compliance (DDC) | 99.539 | 99.444 | 86.267 |
| Traffic light compliance (TLC) | 99.778 | 99.333 | 98.217 |
| Ego progress (EP) | 87.649 | 84.311 | 85.143 |
| Time to collision (TTC) | 97.868 | 96.222 | 77.945 |
| Lane keeping (LK) | 96.987 | 96.667 | 47.076 |
| History comfort (HC) | 98.312 | 97.778 | 96.114 |
| Two-frame extended comfort (EC) | 85.707 | 79.111 | 70.743 |

The larger weaknesses appear in reactive second-stage lane keeping, drivable
area compliance and safety metrics. These describe this checkpoint's behavior;
they do not establish a causal benefit for any untested training change.

## Timing boundary

- Policy starts after image preprocessing and ends at the normalized CPU action.
- E2E starts with in-memory RGB, prepared ego state and cached text, and includes
  preprocessing, CPU-to-GPU transfers, VAE/backbone/head, CPU output and trajectory
  denormalization.
- Scene construction, JPEG decoding, text-cache loading, PDM scoring and file
  output are excluded. This is not sensor-to-actuator latency.
- Main results use uninstrumented synchronized wall time. CUDA-event profiling
  runs in a separate pass. See the [benchmark protocol](../inference_benchmark.md)
  for sampling, warmup and repeat defaults; the exact run settings are in the
  server's resolved configuration.

## Wall-time measurements

All times are milliseconds. Peak memory is PyTorch peak allocated memory, not
total process memory reported by `nvidia-smi`.

| Variant | Policy mean | Policy p50 | Policy p95 | E2E mean | E2E p50 | E2E p95 | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 62.538 | 60.513 | 81.061 | 71.570 | 69.275 | 93.127 | 3.770 |
| early_exit | 54.494 | 52.958 | 63.500 | 63.575 | 62.036 | 75.135 | 3.770 |

Mean E2E latency decreases by **7.995 ms (11.2%)**; policy latency decreases by
**8.044 ms (12.9%)**. The reciprocal E2E rates are 13.97 / 15.73 requests/s,
derived from latency rather than a separate sustained-throughputput measurement.

## Parameters and output parity

| Item | Result |
| --- | ---: |
| Loaded parameters | 1,656,343,350 |
| Trainable parameters by configuration | 110,555,459 |
| Compared scenes | 200 |
| Normalized action tolerance | 0.0 |
| Exact equality on every compared scene | True |
| All comparisons passed | True |
| Maximum normalized absolute error | 0.0 |
| Maximum physical XY distance (m) | 0.0 |
| Maximum wrapped yaw error (rad) | 0.0 |

With zero-indexed taps `[8,16,24]`, `early_exit` executes 25 of the original 30
backbone blocks and skips only computation after the last trajectory tap. It does
not remove weights or reduce resident parameter count: both variants retain the
unused VAE decoder and tail blocks. The unchanged peak allocated memory is
consistent with this execution-only change. Parameter counts are not active
FLOP counts, and trainable-by-configuration is not the total inference model size.

This parity test is evidence for exact trajectory equivalence on the sampled
inputs. It is not a new full NAVSIM score evaluation of `early_exit`. Normal
inference still defaults to the original path; the benchmark explicitly enables
the optional skip for its comparison.

## CUDA event regions

This is a separate profiling pass. Regions are device-timeline elapsed times,
including possible launch gaps, not pure kernel-time sums. Resamplers and decoder
are nested inside `trajectory_head_total`; do not add them to that total again.
Profiled region times should not be reconciled by addition with the wall-time
means from the uninstrumented pass.

| Variant | Region | Mean ms | p50 ms | p95 ms |
| --- | --- | ---: | ---: | ---: |
| baseline | backbone_blocks | 48.812 | 46.387 | 66.048 |
| baseline | backbone_pre | 0.799 | 0.745 | 1.075 |
| baseline | context | 0.466 | 0.439 | 0.607 |
| baseline | resamplers_total | 1.026 | 0.956 | 1.401 |
| baseline | trajectory_decoder | 0.850 | 0.802 | 1.148 |
| baseline | trajectory_head_total | 2.193 | 2.122 | 3.072 |
| baseline | vae_encode | 12.533 | 12.491 | 12.813 |
| early_exit | backbone_blocks | 39.234 | 37.459 | 51.287 |
| early_exit | backbone_pre | 0.734 | 0.721 | 0.786 |
| early_exit | context | 0.490 | 0.416 | 0.497 |
| early_exit | resamplers_total | 1.012 | 0.900 | 1.139 |
| early_exit | trajectory_decoder | 0.779 | 0.752 | 0.889 |
| early_exit | trajectory_head_total | 2.092 | 1.932 | 2.257 |
| early_exit | vae_encode | 12.502 | 12.479 | 12.573 |

The backbone remains the largest measured region, followed by VAE encoding. The
trajectory head takes approximately 2 ms, so changes limited to that head have
little room to reduce this pipeline's latency. Further optimization would need
its own measurements and parity checks.

These results support an internal efficiency comparison for this checkpoint.
They do not establish a speed advantage or a performance–efficiency Pareto
advantage over models measured on different GPUs or with different input and
timing boundaries. No same-hardware external-model baseline is included.

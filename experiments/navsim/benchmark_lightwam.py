"""Batch-one, single-GPU LightWAM-AD timing on real NAVSIM-v2 images.

No metric cache or PDM workers are required. Data loading is outside all timers.
Use run_benchmark_lightwam.sh; --cfg job --resolve inspects config without NAVSIM.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "navsim_v2"):
    sys.path.insert(0, str(path))

from simwam.utils.inference_benchmark import (  # noqa: E402
    CudaStageRecorder, compare_actions, parameter_counts, select_tokens, summarize_ms,
)

LOGGER = logging.getLogger(__name__)
TIMING_KEYS = ("preprocess_ms", "policy_ms", "postprocess_ms", "end_to_end_ms")


def _command(args):
    try:
        return subprocess.check_output(
            args, cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT, timeout=15
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def _gpu_snapshot():
    return _command([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu,"
        "temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv",
    ])


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_options(cfg):
    bench = cfg.BENCHMARK
    for key in ("num_samples", "warmup", "repeats", "cpu_threads"):
        if int(bench[key]) <= 0:
            raise ValueError(f"BENCHMARK.{key} must be positive.")
    if not 0 <= int(bench.profile_samples) <= int(bench.num_samples):
        raise ValueError("profile_samples must be between 0 and num_samples.")
    variants = list(bench.variants)
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("Provide distinct benchmark variants.")
    if set(variants) - {"baseline", "early_exit"} or "baseline" not in variants:
        raise ValueError("variants must include baseline, optionally early_exit.")
    if not np.isfinite(float(bench.parity_atol)) or float(bench.parity_atol) < 0:
        raise ValueError("parity_atol must be finite and nonnegative.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run with Python, not distributed torchrun.")
    if cfg.get("ckpt") is None or not Path(str(cfg.ckpt)).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.get('ckpt')}")


def _load_samples(cfg, v2, loader, tokens):
    data = cfg.data.train
    cache = {}
    samples = []
    for index, token in enumerate(tokens):
        scene = loader.get_scene_from_token(token)
        history = int(scene.scene_metadata.num_history_frames)
        if not 1 <= history <= len(scene.frames):
            raise ValueError(f"Invalid current frame for {token}.")
        image = scene.frames[history - 1].cameras.cam_f0.image
        if image is None:
            raise ValueError(f"Missing front image for {token}; refusing biased sample skipping.")
        prompt = v2._build_prompt_for_scene(scene, history, bool(data.use_dynamic_prompt))
        if prompt not in cache:
            cache[prompt] = v2._load_text_context(
                str(data.text_embedding_cache_dir), int(data.context_len), prompt,
                text_encoder_id=str(data.text_encoder_id),
            )
        context, mask = cache[prompt]
        samples.append({
            "token": token, "image": np.array(image, dtype=np.uint8, copy=True),
            "proprio": v2._extract_proprio(scene).cpu(),
            "context": context.cpu(), "context_mask": mask.cpu(),
        })
        del scene
        if (index + 1) % 20 == 0:
            gc.collect()
            LOGGER.info("Preloaded %d/%d current RGB frames", index + 1, len(tokens))
    return samples


def _request(model, sample, cfg, v2, variant):
    """Matched to predict_navsim_v2.predict, without scene/disk access or np.save."""
    data = cfg.data.train
    torch.cuda.synchronize()
    start = time.perf_counter()
    image = v2._preprocess_front_image(sample["image"], list(data.video_size))
    prepared = time.perf_counter()
    pred = model.infer_action(
        prompt=None, input_image=image,
        action_horizon=int(data.future_action_horizon),
        proprio=sample["proprio"], context=sample["context"],
        context_mask=sample["context_mask"],
        skip_unused_tail=(variant == "early_exit"),
    )["action"]
    torch.cuda.synchronize()
    inferred = time.perf_counter()
    poses = v2._denormalize_action(
        pred, str(data.trajectory_mode), bool(data.normalize_action)
    ).numpy().astype(np.float32)
    finished = time.perf_counter()
    if poses.shape != (int(data.future_action_horizon), 3) or not np.isfinite(poses).all():
        raise ValueError(f"Invalid trajectory for {sample['token']}.")
    return {
        "preprocess_ms": (prepared - start) * 1000,
        "policy_ms": (inferred - prepared) * 1000,
        "postprocess_ms": (finished - inferred) * 1000,
        "end_to_end_ms": (finished - start) * 1000,
    }, pred, poses


def _check_parity(model, samples, cfg, v2, out_dir):
    if "early_exit" not in cfg.BENCHMARK.variants:
        return {"checked": False}
    rows, baseline_poses, early_poses = [], [], []
    for sample in samples:
        _, reference, poses = _request(model, sample, cfg, v2, "baseline")
        _, candidate, optimized = _request(model, sample, cfg, v2, "early_exit")
        check = compare_actions(reference, candidate, float(cfg.BENCHMARK.parity_atol))
        diff = poses.astype(np.float64) - optimized.astype(np.float64)
        yaw = np.arctan2(np.sin(diff[:, 2]), np.cos(diff[:, 2]))
        rows.append({
            "token": sample["token"], **check,
            "max_xy_distance_m": float(np.linalg.norm(diff[:, :2], axis=-1).max()),
            "max_yaw_error_rad": float(np.abs(yaw).max()),
        })
        baseline_poses.append(poses)
        early_poses.append(optimized)
    with (out_dir / "parity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out_dir / "parity_predictions.npz", tokens=np.array([s["token"] for s in samples]),
        baseline=np.stack(baseline_poses), early_exit=np.stack(early_poses),
    )
    result = {
        "checked": True, "num_scenes": len(rows),
        "atol_normalized": float(cfg.BENCHMARK.parity_atol),
        "exact_all": all(row["exact"] for row in rows),
        "passed_all": all(row["passed"] for row in rows),
        "max_abs_error_normalized": max(row["max_abs_error"] for row in rows),
        "max_xy_distance_m": max(row["max_xy_distance_m"] for row in rows),
        "max_yaw_error_rad": max(row["max_yaw_error_rad"] for row in rows),
    }
    _write_json(out_dir / "parity.json", result)
    if not result["passed_all"]:
        raise RuntimeError("Early-exit parity failed; see parity.csv. Comparative timing aborted.")
    LOGGER.info("Parity across all selected scenes: %s", result)
    return result


def _profile_targets(model):
    return [
        ("context", model, "_prepare_inference_context"),
        ("vae_encode", model, "_encode_input_image_latents_tensor"),
        ("backbone_pre", model.video_expert, "pre_dit"),
        ("backbone_blocks", model, "_forward_backbone_once"),
        ("trajectory_head_total", model.trajectory_head, "forward"),
        *[("resamplers_total", module, "forward") for module in model.trajectory_head.resamplers],
        ("trajectory_decoder", model.trajectory_head.decoder, "forward"),
    ]


def _render_summary(summary):
    lines = [
        "# LightWAM-AD inference benchmark", "",
        f"GPU: {summary['environment']['gpu_name']}; batch=1; dtype={summary['environment']['dtype']}",
        f"Checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        f"Git commit: `{summary['git_commit']}`; dirty={summary['git_dirty']}", "",
        "Times exclude scene/JPEG/text-cache loading, PDM scoring and file output.",
        "E2E starts with in-memory RGB + ego state + cached text; includes preprocessing,",
        "CPU-to-GPU transfers, VAE/backbone/head, CPU output and trajectory denormalization.",
        "Policy starts after image preprocessing and ends at the normalized CPU action.",
        "Uninstrumented wall times below; profiling is a separate pass. No torch.compile or LoRA merge.", "",
        "| Variant | Policy mean ms | Policy p50 | Policy p95 | E2E mean ms | E2E p50 | E2E p95 | Peak allocated GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in summary["variants"].items():
        p, e = result["policy_ms"], result["end_to_end_ms"]
        lines.append(
            f"| {name} | {p['mean_ms']:.3f} | {p['p50_ms']:.3f} | {p['p95_ms']:.3f} "
            f"| {e['mean_ms']:.3f} | {e['p50_ms']:.3f} | {e['p95_ms']:.3f} "
            f"| {result['peak_allocated_bytes'] / 2**30:.3f} |"
        )
    lines += ["", f"Parameters: {summary['parameters']['total_loaded']:,} loaded; "
              f"{summary['parameters']['trainable_by_config']:,} trainable by config.",
              "Both variants retain the same loaded weights, including the unused VAE decoder and tail blocks.",
              "Early exit skips execution only; parameter count is not an active-FLOP count.",
              "", f"Parity: `{json.dumps(summary['parity'])}`", "", "## CUDA event regions", "",
              "Separate profiling pass; regions are not pure kernel sums. Resamplers and decoder are nested",
              "inside trajectory_head_total: do not add these nested times to the head total.", "",
              "| Variant | Region | Mean ms | p50 ms | p95 ms |", "|---|---|---:|---:|---:|"]
    for name, regions in summary["cuda_profile"].items():
        for region, stats in regions.items():
            lines.append(f"| {name} | {region} | {stats['mean_ms']:.3f} | {stats['p50_ms']:.3f} | {stats['p95_ms']:.3f} |")
    lines += ["", "Per-repeat measurements, raw timings, scene tokens and GPU snapshots accompany this report.",
              "Sampled parity is not a new full NAVSIM score evaluation.", ""]
    return "\n".join(lines)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="benchmark_navsim")
@torch.no_grad()
def main(cfg: DictConfig):
    _validate_options(cfg)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU, e.g. CUDA_VISIBLE_DEVICES=0.")
    if str(cfg.EVALUATION.device) not in {"cuda", "cuda:0"}:
        raise ValueError("Use logical device cuda:0 with CUDA_VISIBLE_DEVICES selecting the physical GPU.")
    torch.cuda.set_device(0)
    torch.set_num_threads(int(cfg.BENCHMARK.cpu_threads))
    torch.set_num_interop_threads(1)
    random.seed(int(cfg.BENCHMARK.seed))
    np.random.seed(int(cfg.BENCHMARK.seed))
    torch.manual_seed(int(cfg.BENCHMARK.seed))
    torch.cuda.manual_seed_all(int(cfg.BENCHMARK.seed))

    # Defer NAVSIM imports so Hydra config inspection works without the driving dependencies.
    import predict_navsim_v2 as v2
    from simwam.models.wan22.lightwam_ad import LightWAMAD

    v2._assert_navsim_v2()
    v2._ensure_eval_defaults(cfg)
    out_dir = Path(str(cfg.BENCHMARK.output_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"Use an empty benchmark output directory: {out_dir}")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "benchmark.log")],
        force=True,
    )
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)
    _write_json(out_dir / "status.json", {"status": "running"})
    try:
        checkpoint = Path(str(cfg.ckpt)).resolve()
        LOGGER.info("Hashing checkpoint and recording environment; outside latency measurement.")
        checkpoint_hash = _sha256(checkpoint)
        gpu_before = _gpu_snapshot()
        dtype = v2._mixed_precision_to_model_dtype(str(cfg.mixed_precision))
        model = instantiate(cfg.model, model_dtype=dtype, device="cuda:0")
        if not isinstance(model, LightWAMAD):
            raise TypeError("This benchmark supports LightWAMAD only.")
        v2._load_model_checkpoint(model, str(checkpoint))
        # Inference factories do not apply the trainer's requires_grad policy.
        # Apply it for an honest trainable-by-config count, then restore eval mode.
        model.configure_trainable_modules()
        model.eval()
        if "early_exit" in cfg.BENCHMARK.variants and model.tap_layers[-1] == len(model.video_expert.blocks) - 1:
            raise ValueError("This tap configuration has no unused tail; use variants=[baseline].")
        parameters = parameter_counts(model)
        loader = v2._build_scene_loader(cfg)
        tokens = select_tokens(loader.tokens, int(cfg.BENCHMARK.num_samples), int(cfg.BENCHMARK.seed))
        _write_json(out_dir / "tokens.json", tokens)
        LOGGER.info("Preloading %d distinct real scene inputs, no future image or latent cache.", len(tokens))
        samples = _load_samples(cfg, v2, loader, tokens)
        del loader
        gc.collect()
        parity = _check_parity(model, samples, cfg, v2, out_dir)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # Once before benchmark rounds, never inside a timed request.

        rows, rounds, gpu_rounds = [], [], []
        with (out_dir / "latency_samples.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["variant", "repeat", "token", *TIMING_KEYS])
            writer.writeheader()
            for repeat in range(int(cfg.BENCHMARK.repeats)):
                variants = list(cfg.BENCHMARK.variants)
                if repeat % 2:
                    variants.reverse()  # Reduce systematic first/last-run bias.
                order = list(range(len(samples)))
                random.Random(int(cfg.BENCHMARK.seed) + repeat).shuffle(order)
                for variant in variants:
                    LOGGER.info("Warmup: variant=%s repeat=%d count=%d", variant, repeat + 1, cfg.BENCHMARK.warmup)
                    for index in range(int(cfg.BENCHMARK.warmup)):
                        _request(model, samples[order[index % len(order)]], cfg, v2, variant)
                    gc.collect()
                    torch.cuda.synchronize()
                    start_allocated = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    round_rows = []
                    LOGGER.info("Uninstrumented timing: %s repeat=%d", variant, repeat + 1)
                    for index in order:
                        timing, _, _ = _request(model, samples[index], cfg, v2, variant)
                        row = {"variant": variant, "repeat": repeat + 1, "token": samples[index]["token"], **timing}
                        writer.writerow(row)
                        round_rows.append(row)
                    handle.flush()
                    rows.extend(round_rows)
                    rounds.append({
                        "variant": variant, "repeat": repeat + 1,
                        "allocated_before_bytes": start_allocated,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        **{key: summarize_ms([row[key] for row in round_rows]) for key in TIMING_KEYS},
                    })
                    gpu_rounds.append({"variant": variant, "repeat": repeat + 1, "snapshot": _gpu_snapshot()})
                    LOGGER.info("%s: policy %.3f ms; E2E %.3f ms", variant,
                                rounds[-1]["policy_ms"]["mean_ms"], rounds[-1]["end_to_end_ms"]["mean_ms"])

        profile = {}
        profile_rows = []
        for variant in cfg.BENCHMARK.variants:
            recorder = CudaStageRecorder()
            if int(cfg.BENCHMARK.profile_samples):
                with recorder.instrument(_profile_targets(model)):
                    # Discard one instrumented warmup, distinct from headline timing.
                    _request(model, samples[0], cfg, v2, variant)
                    for sample in samples[:int(cfg.BENCHMARK.profile_samples)]:
                        recorder.clear()
                        _request(model, sample, cfg, v2, variant)
                        torch.cuda.synchronize()
                        for region, elapsed in recorder.elapsed_ms().items():
                            profile_rows.append({"variant": variant, "token": sample["token"], "region": region, "cuda_ms": elapsed})
            regions = sorted({row["region"] for row in profile_rows if row["variant"] == variant})
            profile[variant] = {
                region: summarize_ms([row["cuda_ms"] for row in profile_rows
                                      if row["variant"] == variant and row["region"] == region])
                for region in regions
            }
        with (out_dir / "cuda_profile.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["variant", "token", "region", "cuda_ms"])
            writer.writeheader()
            writer.writerows(profile_rows)

        props = torch.cuda.get_device_properties(0)
        summary = {
            "schema_version": 1, "status": "complete", "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "git_commit": _command(["git", "rev-parse", "HEAD"]),
            "git_dirty": bool(_command(["git", "status", "--porcelain", "--untracked-files=no"])),
            "environment": {
                "gpu_name": props.name, "gpu_total_memory_bytes": props.total_memory,
                "compute_capability": list(torch.cuda.get_device_capability()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "dtype": str(dtype),
                "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
                "nvidia_smi_before": gpu_before, "nvidia_smi_after": _gpu_snapshot(),
                "nvidia_smi_rounds": gpu_rounds,
            },
            "protocol": {
                "batch_size": 1, "num_distinct_scenes": len(samples),
                "repeats": int(cfg.BENCHMARK.repeats), "warmup_per_round": int(cfg.BENCHMARK.warmup),
                "video_size_hw": list(cfg.data.train.video_size),
                "scene_filter": str(cfg.EVALUATION.scene_filter), "seed": int(cfg.BENCHMARK.seed),
                "inputs": "in-memory current RGB, CPU ego state and cached text; no cached visual latents",
                "excludes": ["disk/JPEG loading", "scene construction", "text cache loading", "PDM scoring", "file writes"],
                "policy_ms": "CPU preprocessed image through normalized CPU action, including transfers and VAE",
                "end_to_end_ms": "in-memory RGB through CPU float32 physical trajectory, including image preprocessing",
                "profile": "separate CUDA-event regions; head contains resampler and decoder subregions",
                "execution_mode": "eval + no_grad, matching v2 prediction; no compile or LoRA merge",
                "executed_blocks": {
                    name: model.tap_layers[-1] + 1 if name == "early_exit" else len(model.video_expert.blocks)
                    for name in cfg.BENCHMARK.variants
                },
            },
            "parameters": parameters, "parity": parity, "rounds": rounds,
            "cuda_profile": profile,
            "variants": {
                name: {
                    **{key: summarize_ms([row[key] for row in rows if row["variant"] == name]) for key in TIMING_KEYS},
                    "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rounds if r["variant"] == name),
                    "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in rounds if r["variant"] == name),
                } for name in cfg.BENCHMARK.variants
            },
        }
        _write_json(out_dir / "summary.json", summary)
        report = _render_summary(summary)
        (out_dir / "summary.md").write_text(report)
        _write_json(out_dir / "status.json", {"status": "complete"})
        print(report, flush=True)
        print(f"RESULT_DIR={out_dir}", flush=True)
    except Exception as error:
        _write_json(out_dir / "status.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()

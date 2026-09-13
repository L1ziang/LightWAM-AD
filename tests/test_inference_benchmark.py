"""CPU checks for timing bookkeeping and the real model's optional tail skip."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.utils.checkpoint
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from simwam.models.wan22.lightwam_ad import LightWAMAD
from simwam.models.wan22.lora import LIGHTWAM_VIDEO_BLOCK_LORA_TARGETS, apply_lora_to_paths
from simwam.models.wan22.wan_video_dit import WanVideoDiT
from simwam.utils.inference_benchmark import (
    CudaStageRecorder, compare_actions, parameter_counts, select_tokens, summarize_ms,
)

ROOT = Path(__file__).resolve().parents[1]


class DummyVAE(nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 8

    def encode(self, videos, device, tiled=False):
        batch = videos if isinstance(videos, torch.Tensor) else torch.stack(videos)
        batch = batch.to(device)
        return torch.cat([batch, batch.mean(dim=1, keepdim=True)], dim=1)


def _model(checkpointing):
    video = WanVideoDiT(
        hidden_dim=24, in_dim=4, out_dim=4, ffn_dim=48, text_dim=16, freq_dim=16,
        patch_size=(1, 2, 2), num_heads=2, attn_head_dim=12, num_layers=4, eps=1e-6,
        has_image_input=False, seperated_timestep=True,
        require_vae_embedding=False, require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True, action_conditioned=False,
        video_attention_mask_mode="first_frame_causal", use_gradient_checkpointing=checkpointing,
    )
    for block in video.blocks:
        apply_lora_to_paths(block, LIGHTWAM_VIDEO_BLOCK_LORA_TARGETS, r=2, alpha=4.0)
    model = LightWAMAD(
        video_expert=video, vae=DummyVAE(), text_encoder=None, tokenizer=None,
        text_dim=16, proprio_dim=8, action_dim=3, tap_layers=[0, 2], adapter_dim=8,
        trajectory_head_config={"num_slots": 2, "slot_dim": 16, "num_heads": 4,
                                "decoder_layers": 1, "max_action_horizon": 8},
    )
    model.configure_trainable_modules()
    return model


def _kwargs():
    return dict(prompt=None, input_image=torch.randn(1, 3, 16, 16),
                context=torch.randn(5, 16), context_mask=torch.ones(5, dtype=torch.bool),
                proprio=torch.randn(8), action_horizon=8)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_tail_skip_preserves_real_model_output_and_does_not_change_default(checkpointing):
    torch.manual_seed(42)
    model = _model(checkpointing).eval()
    kwargs = _kwargs()
    counts = [0] * 4

    def hook(index):
        def record(*_):
            counts[index] += 1
        return record

    handles = [block.register_forward_hook(hook(i)) for i, block in enumerate(model.video_expert.blocks)]
    baseline = model.infer_action(**kwargs)["action"]
    assert counts == [1, 1, 1, 1]
    before = parameter_counts(model)
    early = model.infer_action(**kwargs, skip_unused_tail=True)["action"]
    assert counts == [2, 2, 2, 1]
    assert torch.equal(baseline, early)
    assert parameter_counts(model) == before
    default_again = model.infer_action(**kwargs)["action"]
    assert counts == [3, 3, 3, 2]
    assert torch.equal(baseline, default_again)
    for handle in handles:
        handle.remove()


def test_tail_skip_cannot_change_training_or_grad_enabled_backbone():
    model = _model(False).train()
    with pytest.raises(RuntimeError, match="eval mode"):
        model.infer_action(**_kwargs(), skip_unused_tail=True)
    model.eval()
    with pytest.raises(RuntimeError, match="disabled gradients"):
        model._forward_backbone_once({}, stop_after_last_tap=True)


def test_training_still_runs_tail_and_has_video_gradient():
    model = _model(False)
    kwargs = _kwargs()
    sample = dict(video_latents=torch.randn(1, 4, 3, 4, 4),
                  action=torch.randn(1, 8, 3), proprio=kwargs["proprio"].unsqueeze(0),
                  context=kwargs["context"].unsqueeze(0),
                  context_mask=kwargs["context_mask"].unsqueeze(0))
    loss, _ = model.training_loss(sample)
    loss.backward()
    grad = model.video_expert.blocks[-1].self_attn.q.lora_B.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_sampling_is_order_independent_and_never_repeats_scenes():
    tokens = [str(i) for i in range(100)]
    selected = select_tokens(tokens, 20, 42)
    assert selected == select_tokens(list(reversed(tokens)), 20, 42)
    assert len(set(selected)) == 20
    with pytest.raises(ValueError):
        select_tokens(tokens, 101, 42)
    with pytest.raises(ValueError):
        select_tokens(["a", "a"], 1, 42)


def test_summary_percentiles_and_rejects_bad_measurements():
    summary = summarize_ms([1, 2, 3, 4, 5])
    assert summary["mean_ms"] == 3
    assert summary["p50_ms"] == 3
    assert summary["p95_ms"] == pytest.approx(4.8)
    for values in ([], [np.nan], [np.inf], [-1]):
        with pytest.raises(ValueError):
            summarize_ms(values)


def test_parity_rejects_nonfinite_or_changed_actions():
    original = torch.zeros(8, 3)
    changed = original.clone()
    changed[0, 0] = 1e-3
    assert compare_actions(original, original.clone(), 0)["exact"]
    assert not compare_actions(original, changed, 0)["passed"]
    assert compare_actions(original, changed, 0.01)["passed"]
    with pytest.raises(ValueError):
        compare_actions(original, torch.full_like(original, float("nan")), 0)


class FakeEvent:
    def record(self):
        pass

    def elapsed_time(self, end):
        return 2.5


def test_profile_spans_sum_and_restore_descriptors_even_after_exception():
    module = nn.Linear(2, 2)
    original = module.forward
    recorder = CudaStageRecorder(FakeEvent)
    with pytest.raises(RuntimeError, match="deliberate"):
        with recorder.instrument([("linear", module, "forward")]):
            module(torch.ones(1, 2))
            module(torch.ones(1, 2))
            assert recorder.elapsed_ms() == {"linear": 5.0}
            raise RuntimeError("deliberate")
    assert "forward" not in vars(module)
    assert module.forward == original


def _driver():
    spec = importlib.util.spec_from_file_location("lightwam_bench_test", ROOT / "experiments/navsim/benchmark_lightwam.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_request_matches_prediction_boundaries_and_validates_output(monkeypatch):
    driver = _driver()
    monkeypatch.setattr(driver.torch.cuda, "synchronize", lambda: None)
    clock = iter([1.0, 1.01, 1.03, 1.031])
    monkeypatch.setattr(driver, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    seen = {}

    def infer(**kwargs):
        seen.update(kwargs)
        return {"action": torch.ones(8, 3)}

    helpers = SimpleNamespace(_preprocess_front_image=lambda image, size: torch.zeros(1, 3, *size),
                              _denormalize_action=lambda pred, mode, normalize: pred * 2)
    cfg = OmegaConf.create({"data": {"train": {"video_size": [16, 16], "future_action_horizon": 8,
                                               "trajectory_mode": "absolute", "normalize_action": True}}})
    sample = {"image": np.zeros((32, 32, 3), dtype=np.uint8), "token": "a",
              "context": torch.ones(5, 16), "context_mask": torch.ones(5), "proprio": torch.zeros(8)}
    timing, _, poses = driver._request(SimpleNamespace(infer_action=infer), sample, cfg, helpers, "early_exit")
    assert timing["preprocess_ms"] == pytest.approx(10)
    assert timing["policy_ms"] == pytest.approx(20)
    assert timing["postprocess_ms"] == pytest.approx(1)
    assert timing["end_to_end_ms"] == pytest.approx(31)
    assert seen["skip_unused_tail"] and seen["input_image"].shape == (1, 3, 16, 16)
    assert poses.dtype == np.float32 and np.all(poses == 2)


def test_hydra_benchmark_inherits_checkpoint_architecture(monkeypatch, tmp_path):
    for name in ("NAVSIM_LOG_PATH", "NAVSIM_SENSOR_BLOBS_PATH", "NAVSIM_DEVKIT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path))
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="benchmark_navsim", overrides=[
            "BENCHMARK.num_samples=10", "BENCHMARK.profile_samples=2",
            "EVALUATION.output_dir=./unused_eval_output",
        ])
    assert cfg.model.backbone_lora.rank == 64
    assert list(cfg.model.tap_layers) == [8, 16, 24]
    assert list(cfg.data.train.video_size) == [384, 672]
    assert cfg.BENCHMARK.num_samples == 10
    assert cfg.EVALUATION.scene_filter.endswith("scene_filter/navtest.yaml")
    OmegaConf.save(cfg, tmp_path / "config.yaml", resolve=True)


def test_failed_parity_saves_diagnostics_and_aborts(monkeypatch, tmp_path):
    driver = _driver()
    cfg = OmegaConf.create({"BENCHMARK": {"variants": ["baseline", "early_exit"], "parity_atol": 0}})

    def changed_request(model, sample, cfg, v2, variant):
        value = 0.0 if variant == "baseline" else 0.001
        action = torch.full((8, 3), value)
        return {}, action, action.numpy()

    monkeypatch.setattr(driver, "_request", changed_request)
    with pytest.raises(RuntimeError, match="parity failed"):
        driver._check_parity(None, [{"token": "a"}], cfg, None, tmp_path)
    assert (tmp_path / "parity.csv").exists()
    assert (tmp_path / "parity_predictions.npz").exists()
    assert not (tmp_path / "summary.json").exists()

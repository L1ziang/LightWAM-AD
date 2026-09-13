from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import nullcontext

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

from simwam.datasets.navsim.navsim_dataset import NavSimVideoDataset
from simwam.datasets.split import deterministic_token_hash_split
from simwam.models.wan22.lightwam_ad import LightWAMAD, SlotPreservingTrajectoryHead
from simwam.models.wan22.lora import (
    LIGHTWAM_VIDEO_BLOCK_LORA_TARGETS,
    LoRALinear,
    apply_lora_to_paths,
)
from simwam.models.wan22.wan_video_dit import WanVideoDiT
from simwam.trainer_tensorboard import Wan22Trainer


class DummyVAE(nn.Module):
    z_dim = 4
    temporal_downsample_factor = 4
    upsampling_factor = 8

    def encode(self, videos, device, tiled=False):
        assert not tiled
        if isinstance(videos, torch.Tensor):
            batch = videos
        else:
            batch = torch.stack(list(videos), dim=0)
        batch = batch.to(device)
        extra = batch.mean(dim=1, keepdim=True)
        return torch.cat([batch, extra], dim=1)


class DummyTokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens):
        self.tokens = list(tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return {"token": self.tokens[index], "value": index}

    @staticmethod
    def denormalize_action(action):
        return action + 1


class DummyValidationAccelerator:
    device = torch.device("cpu")
    process_index = 0

    @staticmethod
    def autocast():
        return nullcontext()

    @staticmethod
    def reduce(tensor, reduction):
        assert reduction == "sum"
        return tensor


class DummyValidationModel:
    @staticmethod
    def training_loss(sample):
        value = sample["video"].float().mean()
        return value, {"loss_video": float(value), "loss_action": float(value * 2)}


def build_tiny_model(*, use_lora: bool = False, spatial_factor: int = 1) -> LightWAMAD:
    torch.manual_seed(7)
    video_expert = WanVideoDiT(
        hidden_dim=24,
        in_dim=4,
        ffn_dim=48,
        out_dim=4,
        text_dim=16,
        freq_dim=16,
        eps=1.0e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        attn_head_dim=12,
        num_layers=3,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        action_conditioned=False,
        video_attention_mask_mode="first_frame_causal",
        use_gradient_checkpointing=False,
    )
    if use_lora:
        for block in video_expert.blocks:
            apply_lora_to_paths(
                block,
                LIGHTWAM_VIDEO_BLOCK_LORA_TARGETS,
                r=2,
                alpha=4.0,
                dropout=0.0,
            )
    model = LightWAMAD(
        video_expert=video_expert,
        vae=DummyVAE(),
        text_encoder=None,
        tokenizer=None,
        text_dim=16,
        proprio_dim=8,
        action_dim=3,
        tap_layers=(0, 1, 2),
        adapter_dim=8,
        trajectory_head_config={
            "num_slots": 4,
            "slot_dim": 16,
            "num_heads": 4,
            "decoder_layers": 1,
            "max_action_horizon": 8,
        },
        video_latent_spatial_downsample_factor=spatial_factor,
        device="cpu",
        torch_dtype=torch.float32,
    )
    return model


def test_slot_count_is_preserved_until_trajectory_decoder():
    head = SlotPreservingTrajectoryHead(
        video_hidden_dim=32,
        action_dim=3,
        num_tap_layers=3,
        proprio_dim=8,
        num_slots=4,
        slot_dim=16,
        num_heads=4,
        decoder_layers=1,
        max_action_horizon=8,
    )
    states = [torch.randn(2, 12, 32) for _ in range(3)]
    slots = [resampler(state) for resampler, state in zip(head.resamplers, states)]
    assert all(slot.shape == (2, 4, 16) for slot in slots)
    output = head(states, action_horizon=8, proprio=torch.randn(2, 8, 8))
    assert output.shape == (2, 8, 3)


def test_deterministic_token_hash_validation_split_is_exact_and_stable():
    tokens = [f"token-{index:04d}" for index in range(1000)]
    train_a, val_a = deterministic_token_hash_split(
        DummyTokenDataset(tokens), val_fraction=0.01, seed=42
    )
    train_b, val_b = deterministic_token_hash_split(
        DummyTokenDataset(list(reversed(tokens))), val_fraction=0.01, seed=42
    )

    assert len(train_a) == 990
    assert len(val_a) == 10
    assert set(train_a.tokens).isdisjoint(val_a.tokens)
    assert set(train_a.tokens) | set(val_a.tokens) == set(tokens)
    assert set(val_a.tokens) == set(val_b.tokens)
    assert train_a.split_metadata.val_ids_sha256 == val_b.split_metadata.val_ids_sha256
    torch.testing.assert_close(
        val_a.denormalize_action(torch.tensor([1.0])), torch.tensor([2.0])
    )


def test_validation_loss_aggregates_every_sample_with_batch_weighting():
    trainer = object.__new__(Wan22Trainer)
    trainer.val_loader = [
        {"video": torch.tensor([1.0, 3.0]).reshape(2, 1)},
        {"video": torch.tensor([8.0]).reshape(1, 1)},
    ]
    trainer.eval_num_batches = None
    trainer.seed = 42
    trainer.global_step = 100
    trainer.accelerator = DummyValidationAccelerator()

    metrics = trainer._evaluate_validation_loss(DummyValidationModel())
    # First batch mean=2 represents two samples, second batch mean=8 represents one.
    assert metrics["num_samples"] == 3
    assert metrics["val_loss"] == 4
    assert metrics["loss_video"] == 4
    assert metrics["loss_action"] == 8


def test_first_frame_mask_blocks_future_keys():
    model = build_tiny_model()
    mask = model.video_expert.build_video_to_video_mask(
        video_seq_len=12,
        video_tokens_per_frame=4,
        device=torch.device("cpu"),
    )
    assert mask[:4, :4].all()
    assert not mask[:4, 4:].any()
    assert mask[4:, :].all()


def test_full_clip_current_taps_match_single_frame_taps():
    model = build_tiny_model().eval()
    context = torch.randn(1, 5, 16)
    context_mask = torch.ones(1, 5, dtype=torch.bool)
    clip = torch.randn(1, 4, 3, 4, 4)

    full_pre = model.video_expert.pre_dit(
        x=clip,
        timestep=torch.tensor([500.0]),
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    _, full_taps = model._forward_backbone_once(full_pre)

    single_pre = model.video_expert.pre_dit(
        x=clip[:, :, :1],
        timestep=torch.zeros(1),
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    _, single_taps = model._forward_backbone_once(single_pre)
    for full, single in zip(full_taps, single_taps):
        torch.testing.assert_close(full, single, rtol=2.0e-5, atol=2.0e-5)


def test_action_gradient_does_not_reach_future_latents():
    model = build_tiny_model().eval()
    context = torch.randn(1, 5, 16)
    context_mask = torch.ones(1, 5, dtype=torch.bool)
    clip = torch.randn(1, 4, 3, 4, 4, requires_grad=True)
    pre = model.video_expert.pre_dit(
        x=clip,
        timestep=torch.tensor([500.0]),
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    _, taps = model._forward_backbone_once(pre)
    action = model.trajectory_head(
        taps,
        action_horizon=8,
        proprio=torch.randn(1, 8),
    )
    action.sum().backward()
    assert clip.grad is not None
    assert clip.grad[:, :, :1].abs().sum() > 0
    torch.testing.assert_close(
        clip.grad[:, :, 1:],
        torch.zeros_like(clip.grad[:, :, 1:]),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_training_loss_uses_cached_native_latents():
    model = build_tiny_model()
    sample = {
        "video_latents": torch.randn(2, 4, 3, 4, 4),
        "context": torch.randn(2, 5, 16),
        "context_mask": torch.ones(2, 5, dtype=torch.bool),
        "proprio": torch.randn(2, 8, 8),
        "action": torch.randn(2, 8, 3),
        "image_is_pad": torch.zeros(2, 9, dtype=torch.bool),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
    }
    loss, metrics = model(sample)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(metrics) == {"loss_video", "loss_action", "loss_video_raw", "loss_action_raw"}


def test_frozen_backbone_policy_trains_only_peft_and_new_heads():
    model = build_tiny_model(use_lora=True)
    model.configure_trainable_modules()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert any(name.endswith("lora_A") for name in trainable)
    assert any(name.endswith("lora_B") for name in trainable)
    assert any(name.startswith("video_expert.head.") for name in trainable)
    assert any(name.startswith("adapters.") for name in trainable)
    assert any(name.startswith("trajectory_head.") for name in trainable)
    assert any(name.startswith("proprio_encoder.") for name in trainable)
    for name in trainable:
        assert (
            name.startswith("video_expert.head.")
            or name.endswith("lora_A")
            or name.endswith("lora_B")
            or name.startswith("adapters.")
            or name.startswith("trajectory_head.")
            or name.startswith("proprio_encoder.")
        ), name


def test_main_lora_topology_covers_attention_and_ffn_exactly():
    model = build_tiny_model(use_lora=True)
    lora_names = {
        name for name, module in model.video_expert.named_modules()
        if isinstance(module, LoRALinear)
    }
    expected = {
        f"blocks.{layer_index}.{target}"
        for layer_index in range(3)
        for target in LIGHTWAM_VIDEO_BLOCK_LORA_TARGETS
    }
    assert lora_names == expected
    assert len(lora_names) == 30


def test_production_lora_config_matches_lightwam_main_capacity():
    config_path = Path(__file__).parents[1] / "configs/model/lightwam_ad_navsim.yaml"
    text = config_path.read_text(encoding="utf-8")
    assert "rank: 64" in text
    assert "alpha: 128.0" in text
    assert "ffn.0" in text and "ffn.2" in text

    hidden_dim = 1536
    ffn_dim = 8960
    rank = 64
    layers = 30
    attention_lora = layers * 8 * rank * (hidden_dim + hidden_dim)
    ffn_lora = layers * 2 * rank * (hidden_dim + ffn_dim)
    assert attention_lora + ffn_lora == 87_490_560


def test_peft_checkpoint_round_trip():
    source = build_tiny_model(use_lora=True)
    source.configure_trainable_modules()
    with torch.no_grad():
        for parameter in source.parameters():
            if parameter.requires_grad:
                parameter.add_(0.125)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "lightwam_ad.pt"
        source.save_checkpoint(path, step=17)
        restored = build_tiny_model(use_lora=True)
        payload = restored.load_checkpoint(path)

    assert payload["step"] == 17
    for module_name in ("video_expert", "adapters", "trajectory_head", "proprio_encoder"):
        source_module = getattr(source, module_name)
        restored_module = getattr(restored, module_name)
        source_state = source_module.state_dict()
        restored_state = restored_module.state_dict()
        if module_name == "video_expert":
            keys = [
                key
                for key in source_state
                if key.startswith("head.") or key.endswith("lora_A") or key.endswith("lora_B")
            ]
        else:
            keys = list(source_state)
        for key in keys:
            torch.testing.assert_close(source_state[key], restored_state[key])


def test_low_resolution_ablation_is_explicit_and_spatial_only():
    model = build_tiny_model(spatial_factor=2)
    native = torch.randn(1, 4, 3, 8, 12)
    coarse = model._prepare_latents_for_backbone(native)
    assert coarse.shape == (1, 4, 3, 4, 6)


def test_action_inference_is_single_frame_and_non_iterative():
    model = build_tiny_model().eval()
    result = model.infer_action(
        prompt=None,
        context=torch.randn(5, 16),
        context_mask=torch.ones(5, dtype=torch.bool),
        input_image=torch.randn(1, 3, 4, 4),
        action_horizon=8,
        proprio=torch.randn(8),
        num_inference_steps=50,  # accepted only for legacy CLI compatibility
    )
    assert set(result) == {"action"}
    assert result["action"].shape == (8, 3)


def test_inference_casts_float32_image_to_model_dtype_before_vae():
    model = build_tiny_model().eval()
    model.torch_dtype = torch.bfloat16
    model.vae = DummyVAE().to(dtype=torch.bfloat16)

    latents = model._encode_input_image_latents_tensor(
        torch.randn(2, 3, 4, 4, dtype=torch.float32)
    )

    assert latents.dtype == torch.bfloat16
    assert latents.shape == (2, 4, 1, 4, 4)


def test_navsim_v2_evaluation_overrides_are_declared():
    config_dir = Path(__file__).parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="sim_navsim",
            overrides=[
                "task=lightwam_ad_navsim_front_384x672",
                "EVALUATION.openscene_data_root=/tmp/navsim",
                "EVALUATION.navsim_log_path=/tmp/navsim/navsim_logs/test",
                "EVALUATION.original_sensor_path=/tmp/navsim/sensor_blobs/test",
                "EVALUATION.synthetic_sensor_path=/tmp/navsim/navhard_two_stage/sensor_blobs",
                "EVALUATION.synthetic_scenes_path=/tmp/navsim/navhard_two_stage/synthetic_scene_pickles",
                "EVALUATION.max_tokens=32",
                "EVALUATION.overwrite=true",
            ],
        )

    assert cfg.EVALUATION.openscene_data_root == "/tmp/navsim"
    assert cfg.EVALUATION.original_sensor_path == "/tmp/navsim/sensor_blobs/test"
    assert cfg.EVALUATION.synthetic_sensor_path.endswith("/sensor_blobs")
    assert cfg.EVALUATION.synthetic_scenes_path.endswith("/synthetic_scene_pickles")
    assert cfg.EVALUATION.max_tokens == 32
    assert cfg.EVALUATION.overwrite is True


def test_navsim_front_sensor_config_loads_only_video_frames():
    frame_indices = list(range(3, 12))
    sensor_config = NavSimVideoDataset._build_sensor_config(
        "front",
        frame_indices=frame_indices,
    )

    assert sensor_config.cam_f0 == frame_indices
    assert sensor_config.get_sensors_at_iteration(2) == []
    assert sensor_config.get_sensors_at_iteration(3) == ["cam_f0"]
    assert sensor_config.get_sensors_at_iteration(11) == ["cam_f0"]
    assert sensor_config.get_sensors_at_iteration(12) == []


def test_navsim_stitched_sensor_config_uses_same_sparse_indices():
    sensor_config = NavSimVideoDataset._build_sensor_config(
        "stitched_front",
        frame_indices=[11, 3, 4, 4],
    )

    assert sensor_config.cam_f0 == [3, 4, 11]
    assert sensor_config.cam_l0 == [3, 4, 11]
    assert sensor_config.cam_r0 == [3, 4, 11]
    assert sensor_config.get_sensors_at_iteration(2) == []
    assert sensor_config.get_sensors_at_iteration(4) == ["cam_f0", "cam_l0", "cam_r0"]
    assert sensor_config.get_sensors_at_iteration(12) == []

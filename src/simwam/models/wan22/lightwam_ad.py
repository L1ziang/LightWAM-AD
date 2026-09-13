from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from simwam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .lora import (
    LIGHTWAM_VIDEO_BLOCK_LORA_TARGETS,
    LoRALinear,
    apply_lora_to_paths,
)
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


logger = get_logger(__name__)


class ResidualAdapter(nn.Module):
    """Token-wise residual adapter shared by video and direct-action supervision."""

    def __init__(self, hidden_dim: int, adapter_dim: int, eps: float = 1.0e-6):
        super().__init__()
        if hidden_dim <= 0 or adapter_dim <= 0:
            raise ValueError("`hidden_dim` and `adapter_dim` must be positive.")
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.act = nn.GELU(approximate="tanh")
        self.up = nn.Linear(adapter_dim, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.up(self.act(self.down(self.norm(tokens))))


class SlotResampler(nn.Module):
    """Compress spatial video tokens while preserving every learned query slot."""

    def __init__(
        self,
        input_dim: int,
        slot_dim: int,
        num_slots: int,
        num_heads: int,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        if input_dim <= 0 or slot_dim <= 0 or num_slots <= 0 or num_heads <= 0:
            raise ValueError("Resampler dimensions, slot count, and head count must be positive.")
        if slot_dim % num_heads != 0:
            raise ValueError(
                f"`slot_dim` must be divisible by `num_heads`, got {slot_dim} and {num_heads}."
            )
        self.num_slots = int(num_slots)
        self.input_norm = nn.LayerNorm(input_dim, eps=eps)
        self.input_proj = nn.Linear(input_dim, slot_dim)
        self.slots = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(slot_dim, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(slot_dim, eps=eps)
        self.ffn_norm = nn.LayerNorm(slot_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(slot_dim * 4, slot_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"SlotResampler expects [B,N,D], got {tuple(tokens.shape)}")
        memory = self.input_proj(self.input_norm(tokens))
        query = self.slots.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        update, _ = self.cross_attn(
            query=self.cross_norm(query),
            key=memory,
            value=memory,
            need_weights=False,
        )
        slots = query + update
        return slots + self.ffn(self.ffn_norm(slots))


class SlotPreservingTrajectoryHead(nn.Module):
    """Direct waypoint decoder over unmerged multi-layer resampler slots."""

    def __init__(
        self,
        video_hidden_dim: int,
        action_dim: int,
        num_tap_layers: int,
        proprio_dim: Optional[int],
        num_slots: int = 8,
        slot_dim: int = 512,
        num_heads: int = 8,
        decoder_layers: int = 2,
        max_action_horizon: int = 64,
        proprio_mean: Optional[Sequence[float]] = None,
        proprio_std: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        if action_dim <= 0 or num_tap_layers <= 0 or decoder_layers <= 0:
            raise ValueError("Action dimension, tap count, and decoder depth must be positive.")
        if max_action_horizon <= 0:
            raise ValueError("`max_action_horizon` must be positive.")
        self.action_dim = int(action_dim)
        self.num_tap_layers = int(num_tap_layers)
        self.max_action_horizon = int(max_action_horizon)
        self.resamplers = nn.ModuleList(
            [
                SlotResampler(
                    input_dim=video_hidden_dim,
                    slot_dim=slot_dim,
                    num_slots=num_slots,
                    num_heads=num_heads,
                )
                for _ in range(num_tap_layers)
            ]
        )
        self.layer_embeddings = nn.Parameter(torch.randn(num_tap_layers, slot_dim) * 0.02)

        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            mean = torch.zeros(self.proprio_dim, dtype=torch.float32)
            std = torch.ones(self.proprio_dim, dtype=torch.float32)
            if proprio_mean is not None:
                mean = torch.as_tensor(proprio_mean, dtype=torch.float32)
            if proprio_std is not None:
                std = torch.as_tensor(proprio_std, dtype=torch.float32)
            if mean.numel() != self.proprio_dim or std.numel() != self.proprio_dim:
                raise ValueError("Proprio normalization statistics must match `proprio_dim`.")
            if torch.any(std <= 0):
                raise ValueError("Every proprio std value must be positive.")
            self.register_buffer("proprio_mean", mean.reshape(1, -1), persistent=True)
            self.register_buffer("proprio_std", std.reshape(1, -1), persistent=True)
            self.proprio_encoder = nn.Sequential(
                nn.Linear(self.proprio_dim, slot_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(slot_dim, slot_dim),
                nn.LayerNorm(slot_dim),
            )
        else:
            self.register_buffer("proprio_mean", torch.empty(1, 0), persistent=False)
            self.register_buffer("proprio_std", torch.empty(1, 0), persistent=False)
            self.proprio_encoder = None

        self.action_queries = nn.Parameter(torch.randn(max_action_horizon, slot_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=slot_dim,
            nhead=num_heads,
            dim_feedforward=slot_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(decoder_layers),
            norm=nn.LayerNorm(slot_dim),
        )
        self.output = nn.Linear(slot_dim, self.action_dim)

    def _encode_proprio(
        self,
        proprio: Optional[torch.Tensor],
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.proprio_encoder is None:
            return None
        if proprio is None:
            raise ValueError("Current ego state is required when `proprio_dim` is configured.")
        if proprio.ndim == 3:
            proprio = proprio[:, 0]
        elif proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or proprio.shape != (batch_size, self.proprio_dim):
            raise ValueError(
                f"Expected current proprio shape ({batch_size}, {self.proprio_dim}), got {tuple(proprio.shape)}."
            )
        normalized = (
            proprio.to(device=device, dtype=torch.float32) - self.proprio_mean.to(device=device)
        ) / self.proprio_std.to(device=device)
        return self.proprio_encoder(normalized.to(dtype=dtype)).unsqueeze(1)

    def forward(
        self,
        layer_states: Sequence[torch.Tensor],
        *,
        action_horizon: int,
        proprio: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if len(layer_states) != self.num_tap_layers:
            raise ValueError(
                f"Expected {self.num_tap_layers} tapped layers, got {len(layer_states)}."
            )
        if action_horizon <= 0 or action_horizon > self.max_action_horizon:
            raise ValueError(
                f"`action_horizon` must be in [1,{self.max_action_horizon}], got {action_horizon}."
            )
        slots = []
        for layer_idx, (resampler, state) in enumerate(zip(self.resamplers, layer_states)):
            slot = resampler(state)
            slots.append(slot + self.layer_embeddings[layer_idx].view(1, 1, -1))
        memory = torch.cat(slots, dim=1)
        batch_size = memory.shape[0]
        state_token = self._encode_proprio(
            proprio,
            batch_size=batch_size,
            dtype=memory.dtype,
            device=memory.device,
        )
        if state_token is not None:
            memory = torch.cat([memory, state_token], dim=1)
        queries = self.action_queries[:action_horizon].unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        return self.output(decoded)


class LightWAMAD(nn.Module):
    """One-pass world-action model with native-resolution video and direct trajectory losses."""

    supports_video_rollout = False

    def __init__(
        self,
        *,
        video_expert: nn.Module,
        vae: nn.Module,
        text_encoder: Optional[nn.Module],
        tokenizer,
        text_dim: int,
        proprio_dim: Optional[int],
        action_dim: int,
        tap_layers: Sequence[int],
        adapter_dim: int,
        trajectory_head_config: Optional[dict[str, Any]] = None,
        freeze_backbone: bool = True,
        video_latent_spatial_downsample_factor: int = 1,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        action_loss_beta: float = 0.1,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.action_dim = int(action_dim)
        self.freeze_backbone = bool(freeze_backbone)
        self.video_latent_spatial_downsample_factor = int(video_latent_spatial_downsample_factor)
        if self.video_latent_spatial_downsample_factor < 1:
            raise ValueError("`video_latent_spatial_downsample_factor` must be >= 1.")
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.action_loss_beta = float(action_loss_beta)
        if self.loss_lambda_video < 0 or self.loss_lambda_action < 0:
            raise ValueError("Loss weights must be non-negative.")
        if self.loss_lambda_video == 0 and self.loss_lambda_action == 0:
            raise ValueError("At least one training loss must have non-zero weight.")
        if self.action_loss_beta <= 0:
            raise ValueError("`action_loss_beta` must be positive.")

        num_layers = len(self.video_expert.blocks)
        normalized_taps = tuple(sorted({int(index) for index in tap_layers}))
        if not normalized_taps or normalized_taps[0] < 0 or normalized_taps[-1] >= num_layers:
            raise ValueError(f"Invalid tap layers {normalized_taps} for {num_layers} video layers.")
        self.tap_layers = normalized_taps
        self.adapters = nn.ModuleDict(
            {
                str(index): ResidualAdapter(
                    hidden_dim=int(self.video_expert.hidden_dim),
                    adapter_dim=int(adapter_dim),
                )
                for index in self.tap_layers
            }
        )
        head_config = {} if trajectory_head_config is None else dict(trajectory_head_config)
        self.trajectory_head = SlotPreservingTrajectoryHead(
            video_hidden_dim=int(self.video_expert.hidden_dim),
            action_dim=self.action_dim,
            num_tap_layers=len(self.tap_layers),
            proprio_dim=self.proprio_dim,
            **head_config,
        )
        self.proprio_encoder = (
            nn.Linear(self.proprio_dim, self.text_dim) if self.proprio_dim is not None else None
        )
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.to(device=self.device, dtype=self.torch_dtype)
        self.vae.eval().requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.eval().requires_grad_(False)

    @property
    def dit(self):
        """Compatibility alias used by the original trainer and evaluation code."""
        return self.video_expert

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str,
        tokenizer_model_id: str,
        video_backbone_type: str,
        video_backbone_name: Optional[str],
        video_dit_config: dict[str, Any],
        tokenizer_max_len: int = 128,
        load_text_encoder: bool = False,
        proprio_dim: Optional[int] = 8,
        action_dim: int = 3,
        tap_layers: Sequence[int] = (8, 16, 24),
        adapter_dim: int = 256,
        trajectory_head_config: Optional[dict[str, Any]] = None,
        freeze_backbone: bool = True,
        video_latent_spatial_downsample_factor: int = 1,
        use_backbone_lora: bool = True,
        lora_layer_indices: Optional[Sequence[int]] = None,
        lora_target_modules: Optional[Sequence[str] | str] = None,
        lora_rank: int = 64,
        lora_alpha: float = 128.0,
        lora_dropout: float = 0.0,
        redirect_common_files: bool = False,
        skip_dit_load_from_pretrain: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        action_loss_beta: float = 0.1,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "LightWAMAD":
        from .helpers.loader import apply_video_backbone_preset, load_wan_video_components

        resolved_dit_config = apply_video_backbone_preset(video_dit_config, video_backbone_type)
        components = load_wan_video_components(
            video_backbone_type=video_backbone_type,
            video_backbone_name=video_backbone_name,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            device=device,
            torch_dtype=torch_dtype,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=resolved_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        if int(components.vae.z_dim) != int(components.dit.in_dim):
            raise ValueError(
                f"VAE/DiT channel mismatch: vae.z_dim={components.vae.z_dim}, dit.in_dim={components.dit.in_dim}."
            )
        if int(components.dit.in_dim) != int(resolved_dit_config["in_dim"]):
            raise ValueError("Resolved DiT input channels differ from the instantiated backbone.")

        resolved_lora_indices: tuple[int, ...] = ()
        resolved_lora_targets: tuple[str, ...] = ()
        if use_backbone_lora:
            raw_indices = (
                range(len(components.dit.blocks))
                if lora_layer_indices is None
                else lora_layer_indices
            )
            resolved_lora_indices = tuple(int(index) for index in raw_indices)
            if not resolved_lora_indices:
                raise ValueError("Backbone LoRA is enabled but no layer indices were selected.")
            if len(set(resolved_lora_indices)) != len(resolved_lora_indices):
                raise ValueError(f"Duplicate LoRA layer indices: {resolved_lora_indices}")
            if lora_target_modules is None:
                resolved_lora_targets = LIGHTWAM_VIDEO_BLOCK_LORA_TARGETS
            elif isinstance(lora_target_modules, str):
                resolved_lora_targets = tuple(
                    target.strip() for target in lora_target_modules.split(",")
                )
            else:
                resolved_lora_targets = tuple(
                    str(target).strip() for target in lora_target_modules
                )
            wrapped = 0
            for index in resolved_lora_indices:
                if index < 0 or index >= len(components.dit.blocks):
                    raise ValueError(f"Invalid LoRA layer index {index}.")
                wrapped += apply_lora_to_paths(
                    components.dit.blocks[index],
                    target_paths=resolved_lora_targets,
                    r=int(lora_rank),
                    alpha=float(lora_alpha),
                    dropout=float(lora_dropout),
                )
            logger.info(
                "Applied backbone LoRA to %d projections: layers=%s targets=%s "
                "rank=%d alpha=%.3f dropout=%.3f.",
                wrapped,
                list(resolved_lora_indices),
                list(resolved_lora_targets),
                int(lora_rank),
                float(lora_alpha),
                float(lora_dropout),
            )

        model = cls(
            video_expert=components.dit,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(resolved_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            tap_layers=tap_layers,
            adapter_dim=adapter_dim,
            trajectory_head_config=trajectory_head_config,
            freeze_backbone=freeze_backbone,
            video_latent_spatial_downsample_factor=video_latent_spatial_downsample_factor,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            action_loss_beta=action_loss_beta,
            device=device,
            torch_dtype=torch_dtype,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
        }
        model.backbone_lora_config = {
            "enabled": bool(use_backbone_lora),
            "layer_indices": list(resolved_lora_indices),
            "target_modules": list(resolved_lora_targets),
            "rank": int(lora_rank),
            "alpha": float(lora_alpha),
            "dropout": float(lora_dropout),
        }
        return model

    def configure_trainable_modules(self) -> None:
        self.eval()
        self.requires_grad_(False)
        if not self.freeze_backbone:
            self.video_expert.train().requires_grad_(True)
        else:
            self.video_expert.head.train().requires_grad_(True)
            for module in self.video_expert.modules():
                if isinstance(module, LoRALinear):
                    module.train()
            for name, parameter in self.video_expert.named_parameters():
                if name.endswith("lora_A") or name.endswith("lora_B"):
                    parameter.requires_grad_(True)
        self.adapters.train().requires_grad_(True)
        self.trajectory_head.train().requires_grad_(True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train().requires_grad_(True)

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("Prompt encoding is unavailable; provide cached context tensors.")
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        context = self.text_encoder(ids, mask)
        lengths = mask.gt(0).sum(dim=1).long()
        for batch_idx, length in enumerate(lengths):
            context[batch_idx, length:] = 0
        return context, torch.ones_like(mask)

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None:
            return context, context_mask
        if proprio is None:
            raise ValueError("`proprio` is required by the configured model.")
        if proprio.ndim == 3:
            proprio = proprio[:, 0]
        elif proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if (
            proprio.ndim != 2
            or proprio.shape[0] != context.shape[0]
            or proprio.shape[-1] != self.proprio_dim
        ):
            raise ValueError(f"Expected proprio [B,{self.proprio_dim}], got {tuple(proprio.shape)}")
        token = self.proprio_encoder(
            proprio.to(device=context.device, dtype=context.dtype)
        ).unsqueeze(1)
        mask = torch.ones((context.shape[0], 1), dtype=torch.bool, device=context.device)
        return torch.cat([context, token], dim=1), torch.cat([context_mask, mask], dim=1)

    @torch.no_grad()
    def _encode_video_latents(self, video: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        return self.vae.encode(video, device=self.device, tiled=tiled)

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(f"Expected input image [B,3,H,W], got {tuple(input_image.shape)}")
        input_image = input_image.to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        videos = [image.unsqueeze(1) for image in input_image]
        return self.vae.encode(videos, device=self.device, tiled=tiled)

    def _validate_latents(self, latents: torch.Tensor) -> None:
        expected_channels = int(self.video_expert.in_dim)
        if latents.ndim != 5 or latents.shape[1] != expected_channels:
            raise ValueError(
                f"Expected video latents [B,{expected_channels},T,H,W], got {tuple(latents.shape)}."
            )
        patch_h, patch_w = int(self.video_expert.patch_size[1]), int(self.video_expert.patch_size[2])
        if latents.shape[-2] % patch_h != 0 or latents.shape[-1] % patch_w != 0:
            raise ValueError("Latent spatial dimensions must be divisible by the DiT patch size.")

    def _prepare_latents_for_backbone(self, latents: torch.Tensor) -> torch.Tensor:
        factor = self.video_latent_spatial_downsample_factor
        if factor == 1:
            return latents
        if latents.shape[-2] % factor != 0 or latents.shape[-1] % factor != 0:
            raise ValueError(
                "Native latent grid must be divisible by the optional spatial downsample factor, "
                f"got {tuple(latents.shape[-2:])} and factor={factor}."
            )
        return F.avg_pool3d(
            latents,
            kernel_size=(1, factor, factor),
            stride=(1, factor, factor),
        )

    def build_inputs(self, sample: dict[str, Any], tiled: bool = False) -> dict[str, Any]:
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("Training requires cached `context` and `context_mask`.")
        if "action" not in sample:
            raise ValueError("Training requires a direct trajectory target in `action`.")
        video = sample.get("video")
        if "video_latents" in sample:
            latents = sample["video_latents"].to(self.device, dtype=self.torch_dtype)
        else:
            if video is None or video.ndim != 5:
                raise ValueError("`video` must be [B,3,T,H,W] when no latent cache is provided.")
            if video.shape[1] != 3:
                raise ValueError(f"Raw video must have three RGB channels, got {tuple(video.shape)}.")
            if video.shape[2] % 4 != 1:
                raise ValueError(f"Raw video T must satisfy T % 4 == 1, got {video.shape[2]}.")
            spatial_multiple_h = (
                int(self.vae.upsampling_factor)
                * self.video_latent_spatial_downsample_factor
                * int(self.video_expert.patch_size[1])
            )
            spatial_multiple_w = (
                int(self.vae.upsampling_factor)
                * self.video_latent_spatial_downsample_factor
                * int(self.video_expert.patch_size[2])
            )
            if (
                video.shape[-2] % spatial_multiple_h != 0
                or video.shape[-1] % spatial_multiple_w != 0
            ):
                raise ValueError(
                    "Raw video spatial dimensions must preserve the complete VAE/DiT grid; "
                    f"expected multiples of ({spatial_multiple_h},{spatial_multiple_w}), "
                    f"got {tuple(video.shape[-2:])}."
                )
            latents = self._encode_video_latents(
                video.to(self.device, dtype=self.torch_dtype), tiled=tiled
            )
        latents = self._prepare_latents_for_backbone(latents)
        self._validate_latents(latents)
        if latents.shape[2] <= 1:
            raise ValueError("Training requires at least one future latent frame.")
        batch_size = int(latents.shape[0])

        context = sample["context"].to(self.device, dtype=self.torch_dtype)
        context_mask = sample["context_mask"].to(self.device, dtype=torch.bool)
        if context.ndim != 3 or context.shape[0] != batch_size or context.shape[2] != self.text_dim:
            raise ValueError(
                f"Expected context [B,L,{self.text_dim}] with B={batch_size}, got {tuple(context.shape)}."
            )
        if context_mask.ndim != 2 or context_mask.shape != context.shape[:2]:
            raise ValueError(
                "Expected context_mask to match context [B,L], got "
                f"{tuple(context_mask.shape)} vs {tuple(context.shape[:2])}."
            )
        proprio = sample.get("proprio")
        if proprio is not None and self.proprio_dim is None:
            proprio = None
        elif proprio is not None:
            proprio = proprio.to(self.device, dtype=self.torch_dtype)
            if proprio.ndim not in (2, 3) or proprio.shape[0] != batch_size:
                raise ValueError(
                    f"Expected proprio [B,D] or [B,T,D] with B={batch_size}, got {tuple(proprio.shape)}."
                )
            if proprio.shape[-1] != self.proprio_dim:
                raise ValueError(
                    f"Expected proprio last dimension {self.proprio_dim}, got {proprio.shape[-1]}."
                )
        context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)
        action = sample["action"].to(self.device, dtype=self.torch_dtype)
        if (
            action.ndim != 3
            or action.shape[0] != batch_size
            or action.shape[-1] != self.action_dim
        ):
            raise ValueError(f"Expected action [B,T,{self.action_dim}], got {tuple(action.shape)}")

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            if action_is_pad.ndim != 2 or action_is_pad.shape != action.shape[:2]:
                raise ValueError(
                    "Expected action_is_pad to match action [B,T], got "
                    f"{tuple(action_is_pad.shape)} vs {tuple(action.shape[:2])}."
                )

        image_is_pad = sample.get("image_is_pad")
        if image_is_pad is not None:
            expected_raw_frames = 1 + (latents.shape[2] - 1) * int(
                self.vae.temporal_downsample_factor
            )
            if image_is_pad.ndim != 2 or image_is_pad.shape != (
                batch_size,
                expected_raw_frames,
            ):
                raise ValueError(
                    "Expected image_is_pad to align with the causal VAE latent timeline: "
                    f"({batch_size}, {expected_raw_frames}), got {tuple(image_is_pad.shape)}."
                )
        return {
            "input_latents": latents,
            "context": context,
            "context_mask": context_mask,
            "proprio": proprio,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    def _forward_backbone_once(
        self,
        pre_state: dict[str, Any],
        *,
        stop_after_last_tap: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if stop_after_last_tap and (self.training or torch.is_grad_enabled()):
            raise RuntimeError("Skipping unused blocks requires eval mode and disabled gradients.")
        tokens = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]
        tokens_per_frame = int(pre_state["meta"]["tokens_per_frame"])
        self_attn_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=tokens.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=tokens.device,
        )
        captured = []
        tap_set = set(self.tap_layers)
        for layer_idx, block in enumerate(self.video_expert.blocks):
            tokens = gradient_checkpoint_forward(
                block,
                bool(self.video_expert.use_gradient_checkpointing),
                tokens,
                context,
                t_mod,
                freqs,
                context_mask=context_mask,
                self_attn_mask=self_attn_mask,
            )
            adapter = self.adapters[str(layer_idx)] if str(layer_idx) in self.adapters else None
            if adapter is not None:
                tokens = adapter(tokens)
            if layer_idx in tap_set:
                captured.append(tokens[:, :tokens_per_frame])
            if stop_after_last_tap and layer_idx == self.tap_layers[-1]:
                break
        if len(captured) != len(self.tap_layers):
            raise RuntimeError("Failed to capture all configured action tap layers.")
        return tokens, captured

    def _video_loss_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        token_loss = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return token_loss.mean(dim=1)
        image_is_pad = image_is_pad.to(device=token_loss.device, dtype=torch.bool)
        factor = int(self.vae.temporal_downsample_factor)
        if (image_is_pad.shape[1] - 1) % factor != 0:
            raise ValueError("Raw image padding mask cannot be aligned with VAE latent chunks.")
        latent_pad = image_is_pad[:, 1:].reshape(image_is_pad.shape[0], -1, factor).all(dim=2)
        if latent_pad.shape[1] != token_loss.shape[1]:
            raise ValueError("Future latent padding mask shape mismatch.")
        valid = (~latent_pad).to(dtype=token_loss.dtype)
        return (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _action_loss_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss = F.smooth_l1_loss(
            pred.float(), target.float(), reduction="none", beta=self.action_loss_beta
        ).mean(dim=-1)
        if action_is_pad is None:
            return loss.mean(dim=1)
        valid = (~action_is_pad.to(device=loss.device, dtype=torch.bool)).to(loss.dtype)
        return (loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def training_loss(self, sample: dict[str, Any], tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        clean = inputs["input_latents"]
        noise = torch.randn_like(clean)
        timestep = self.train_video_scheduler.sample_training_t(
            batch_size=clean.shape[0], device=clean.device, dtype=clean.dtype
        )
        noisy = self.train_video_scheduler.add_noise(clean, noise, timestep)
        video_target = self.train_video_scheduler.training_target(clean, noise, timestep)
        noisy[:, :, :1] = clean[:, :, :1]

        pre_state = self.video_expert.pre_dit(
            x=noisy,
            timestep=timestep,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        final_tokens, current_states = self._forward_backbone_once(pre_state)
        pred_video = self.video_expert.post_dit(final_tokens, pre_state)[:, :, 1:]
        video_target = video_target[:, :, 1:]
        pred_action = self.trajectory_head(
            current_states,
            action_horizon=inputs["action"].shape[1],
            proprio=inputs["proprio"],
        )
        video_loss_per_sample = self._video_loss_per_sample(
            pred_video, video_target, inputs["image_is_pad"]
        )
        video_weight = self.train_video_scheduler.training_weight(timestep).to(
            device=video_loss_per_sample.device, dtype=video_loss_per_sample.dtype
        )
        loss_video = (video_loss_per_sample * video_weight).mean()
        loss_action = self._action_loss_per_sample(
            pred_action, inputs["action"], inputs["action_is_pad"]
        ).mean()
        total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach()),
            "loss_video_raw": float(loss_video.detach()),
            "loss_action_raw": float(loss_action.detach()),
        }

    def _prepare_inference_context(
        self,
        *,
        prompt: Optional[Union[str, Sequence[str]]],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if context is None or context_mask is None:
            if prompt is None:
                raise ValueError("Provide either prompt or cached context/context_mask.")
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(self.device, dtype=torch.bool)
        if context.ndim != 3 or context.shape[-1] != self.text_dim:
            raise ValueError(
                f"Expected inference context [B,L,{self.text_dim}], got {tuple(context.shape)}."
            )
        if context_mask.ndim != 2 or context_mask.shape != context.shape[:2]:
            raise ValueError(
                "Inference context_mask must match context [B,L], got "
                f"{tuple(context_mask.shape)} vs {tuple(context.shape[:2])}."
            )
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(self.device, dtype=self.torch_dtype)
        context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)
        return context, context_mask, proprio

    @torch.no_grad()
    def infer_action(
        self,
        *,
        prompt: Optional[Union[str, Sequence[str]]],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        tiled: bool = False,
        skip_unused_tail: bool = False,
        **unused_kwargs,
    ) -> dict[str, torch.Tensor]:
        del unused_kwargs
        context, context_mask, proprio = self._prepare_inference_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        current_latent = self._prepare_latents_for_backbone(
            self._encode_input_image_latents_tensor(input_image, tiled=tiled)
        )
        self._validate_latents(current_latent)
        timestep = torch.zeros(
            current_latent.shape[0], device=self.device, dtype=current_latent.dtype
        )
        pre_state = self.video_expert.pre_dit(
            x=current_latent,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        _, current_states = self._forward_backbone_once(
            pre_state, stop_after_last_tap=skip_unused_tail
        )
        action = self.trajectory_head(
            current_states,
            action_horizon=int(action_horizon),
            proprio=proprio,
        )
        return {"action": action[0].detach().to(device="cpu", dtype=torch.float32)}

    @torch.no_grad()
    def infer_joint(self, **kwargs):
        return self.infer_action(**kwargs)

    @torch.no_grad()
    def infer(self, **kwargs):
        if "num_frames" in kwargs and "num_video_frames" not in kwargs:
            kwargs["num_video_frames"] = kwargs.pop("num_frames")
        kwargs.pop("action", None)
        return self.infer_action(**kwargs)

    def save_checkpoint(self, path, optimizer=None, step=None):
        video_state = self.video_expert.state_dict()
        if self.freeze_backbone:
            video_state = {
                key: value
                for key, value in video_state.items()
                if key.startswith("head.") or key.endswith("lora_A") or key.endswith("lora_B")
            }
        payload = {
            "format": "lightwam_ad_v1",
            "step": step,
            "video_expert": video_state,
            "video_expert_is_peft": self.freeze_backbone,
            "adapters": self.adapters.state_dict(),
            "trajectory_head": self.trajectory_head.state_dict(),
            "tap_layers": list(self.tap_layers),
            "backbone_lora": getattr(self, "backbone_lora_config", None),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") != "lightwam_ad_v1":
            raise ValueError(f"Unsupported LightWAM-AD checkpoint format in {path}.")
        if tuple(payload.get("tap_layers", ())) != self.tap_layers:
            raise ValueError("Checkpoint tap layers differ from the current architecture.")
        checkpoint_lora = payload.get("backbone_lora")
        current_lora = getattr(self, "backbone_lora_config", None)
        if checkpoint_lora is not None and current_lora is not None and checkpoint_lora != current_lora:
            raise ValueError(
                "Checkpoint backbone LoRA topology differs from the current architecture: "
                f"checkpoint={checkpoint_lora}, current={current_lora}."
            )
        video_state = payload["video_expert"]
        is_peft = bool(payload.get("video_expert_is_peft", False))
        if is_peft:
            expected_keys = {
                key
                for key in self.video_expert.state_dict()
                if key.startswith("head.") or key.endswith("lora_A") or key.endswith("lora_B")
            }
            actual_keys = set(video_state)
            if actual_keys != expected_keys:
                raise ValueError(
                    "Checkpoint/current PEFT topology mismatch: "
                    f"missing={sorted(expected_keys - actual_keys)}, "
                    f"unexpected={sorted(actual_keys - expected_keys)}."
                )
            result = self.video_expert.load_state_dict(video_state, strict=False)
            if result.unexpected_keys:
                raise ValueError(
                    f"Unexpected video-expert checkpoint keys: {result.unexpected_keys}"
                )
        else:
            self.video_expert.load_state_dict(video_state, strict=True)
        self.adapters.load_state_dict(payload["adapters"], strict=True)
        self.trajectory_head.load_state_dict(payload["trajectory_head"], strict=True)
        if self.proprio_encoder is not None:
            if "proprio_encoder" not in payload:
                raise ValueError("Checkpoint is missing the configured proprio_encoder.")
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        elif "proprio_encoder" in payload:
            raise ValueError("Checkpoint has a proprio_encoder but the current model disables it.")
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)

import logging
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from .trainer_tensorboard import Wan22Trainer
from .datasets.split import deterministic_token_hash_split
from .utils.logging_config import get_logger, setup_logging
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _as_plain_dict(value, *, name: str, default=None):
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        value = {} if default is None else default
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
    return dict(value)


def create_lightwam_ad(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    video_backbone_type: str = "wan2_1_t2v",
    video_backbone_name: str | None = None,
    tokenizer_max_len: int = 128,
    load_text_encoder: bool = False,
    proprio_dim: int | None = 8,
    action_dim: int = 3,
    tap_layers=(8, 16, 24),
    adapter_dim: int = 256,
    trajectory_head=None,
    freeze_backbone: bool = True,
    video_latent_spatial_downsample_factor: int = 1,
    backbone_lora=None,
    redirect_common_files: bool = False,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    loss=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Hydra factory for the one-pass LightWAM-AD architecture."""
    from .models.wan22.lightwam_ad import LightWAMAD

    video_dit_config = _as_plain_dict(video_dit_config, name="video_dit_config")
    trajectory_head = _as_plain_dict(trajectory_head, name="trajectory_head")
    backbone_lora = _as_plain_dict(backbone_lora, name="backbone_lora")
    video_scheduler = _as_plain_dict(video_scheduler, name="video_scheduler")
    loss = _as_plain_dict(loss, name="loss")

    return LightWAMAD.from_pretrained(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        video_backbone_type=str(video_backbone_type),
        video_backbone_name=video_backbone_name,
        video_dit_config=video_dit_config,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        action_dim=int(action_dim),
        tap_layers=tuple(int(index) for index in tap_layers),
        adapter_dim=int(adapter_dim),
        trajectory_head_config=trajectory_head,
        freeze_backbone=bool(freeze_backbone),
        video_latent_spatial_downsample_factor=int(video_latent_spatial_downsample_factor),
        use_backbone_lora=bool(backbone_lora.get("enabled", True)),
        lora_layer_indices=backbone_lora.get("layer_indices"),
        lora_target_modules=backbone_lora.get("target_modules"),
        lora_rank=int(backbone_lora.get("rank", 64)),
        lora_alpha=float(backbone_lora.get("alpha", 128.0)),
        lora_dropout=float(backbone_lora.get("dropout", 0.0)),
        redirect_common_files=bool(redirect_common_files),
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        action_loss_beta=float(loss.get("action_smooth_l1_beta", 0.1)),
        device=device,
        torch_dtype=model_dtype,
    )


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        split_cfg = data_cfg.get("validation_split")
        if split_cfg is not None and bool(split_cfg.get("enabled", False)):
            strategy = str(split_cfg.get("strategy", "token_hash")).strip().lower()
            if strategy != "token_hash":
                raise ValueError(
                    f"Unsupported validation split strategy `{strategy}`; expected `token_hash`."
                )
            train_ds, val_ds = deterministic_token_hash_split(
                train_ds,
                val_fraction=float(split_cfg.get("val_fraction", 0.01)),
                seed=int(split_cfg.get("seed", 42)),
            )
            metadata = train_ds.split_metadata
            logger.info(
                "Deterministic train/val split: strategy=%s seed=%d fraction=%.6f "
                "total=%d train=%d val=%d val_ids_sha256=%s",
                metadata.strategy,
                metadata.seed,
                metadata.val_fraction,
                metadata.total_size,
                metadata.train_size,
                metadata.val_size,
                metadata.val_ids_sha256,
            )
        else:
            val_ds = None
            logger.info("No validation dataset or validation split is configured.")
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def run_training(cfg: DictConfig):
    misc.register_work_dir(cfg.output_dir)
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
        log_file=Path(cfg.output_dir) / "train.log",
    )
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    
    # ******* build datasets and trainer, then run training *******
    train_ds, val_ds = build_datasets(cfg.data)

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()

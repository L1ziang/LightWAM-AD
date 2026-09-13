from dataclasses import dataclass
import inspect
from typing import Any

import torch
import time

from .io import ModelConfig, hash_model_file, load_state_dict
from .state_dict_converters import (
    wan_video_dit_from_diffusers,
    wan_video_dit_state_dict_converter,
    wan_video_vae_state_dict_converter,
)
from ..wan_video_dit import WanVideoDiT
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..wan_video_vae import WanVideoVAE, WanVideoVAE38
from simwam.utils.logging_config import get_logger

logger = get_logger(__name__)
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"
DEFAULT_VIDEO_BACKBONE_TYPE = "wan2_2_ti2v"


@dataclass
class Wan22LoadedComponents:
    dit: WanVideoDiT
    vae: WanVideoVAE | WanVideoVAE38
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None


@dataclass(frozen=True)
class WanVideoBackboneSpec:
    backbone_type: str
    default_model_id: str
    log_name: str
    dit_origin_file_pattern: str
    text_origin_file_pattern: str
    vae_origin_file_pattern: str
    vae_class: type[torch.nn.Module]
    dit_config_overrides: dict[str, Any]


WAN_VIDEO_BACKBONE_SPECS: dict[str, WanVideoBackboneSpec] = {
    "wan2_2_ti2v": WanVideoBackboneSpec(
        backbone_type="wan2_2_ti2v",
        default_model_id="Wan-AI/Wan2.2-TI2V-5B",
        log_name="Wan2.2-TI2V-5B",
        dit_origin_file_pattern="diffusion_pytorch_model*.safetensors",
        text_origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
        vae_origin_file_pattern="Wan2.2_VAE.pth",
        vae_class=WanVideoVAE38,
        dit_config_overrides={},
    ),
    "wan2_1_t2v": WanVideoBackboneSpec(
        backbone_type="wan2_1_t2v",
        default_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        log_name="Wan2.1-T2V-1.3B",
        dit_origin_file_pattern="diffusion_pytorch_model*.safetensors",
        text_origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
        vae_origin_file_pattern="Wan2.1_VAE.pth",
        vae_class=WanVideoVAE,
        dit_config_overrides={
            "hidden_dim": 1536,
            "in_dim": 16,
            "ffn_dim": 8960,
            "out_dim": 16,
            "patch_size": [1, 2, 2],
            "num_heads": 12,
            "attn_head_dim": 128,
            "num_layers": 30,
            "text_dim": 4096,
            "freq_dim": 256,
            "eps": 1.0e-6,
        },
    ),
}


def resolve_video_backbone_type(video_backbone_type: str | None) -> str:
    resolved = DEFAULT_VIDEO_BACKBONE_TYPE if video_backbone_type is None else str(video_backbone_type).strip().lower()
    if resolved not in WAN_VIDEO_BACKBONE_SPECS:
        raise ValueError(
            f"Unsupported `video_backbone_type`: {video_backbone_type}. "
            f"Expected one of {sorted(WAN_VIDEO_BACKBONE_SPECS)}."
        )
    return resolved


def resolve_video_backbone_name(
    video_backbone_type: str | None,
    video_backbone_name: str | None,
    fallback_model_id: str | None,
) -> str:
    resolved_type = resolve_video_backbone_type(video_backbone_type)
    if video_backbone_name not in (None, ""):
        return str(video_backbone_name)
    if resolved_type == DEFAULT_VIDEO_BACKBONE_TYPE and fallback_model_id not in (None, ""):
        return str(fallback_model_id)
    return WAN_VIDEO_BACKBONE_SPECS[resolved_type].default_model_id


def apply_video_backbone_preset(
    dit_config: dict[str, Any],
    video_backbone_type: str | None,
) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")
    spec = WAN_VIDEO_BACKBONE_SPECS[resolve_video_backbone_type(video_backbone_type)]
    merged = dict(dit_config)
    overridden = {}
    for key, value in spec.dit_config_overrides.items():
        if merged.get(key) != value:
            overridden[key] = (merged.get(key), value)
        merged[key] = value
    if overridden:
        logger.info(
            "Applying `%s` backbone overrides: %s",
            spec.backbone_type,
            ", ".join(f"{key}: {old} -> {new}" for key, (old, new) in sorted(overridden.items())),
        )
    return merged


WAN22_MODEL_REGISTRY = [
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": WanTextEncoder,
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": WanVideoDiT,
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth")
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": WanVideoVAE38,
        "state_dict_converter": wan_video_vae_state_dict_converter,
    },
]


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")

    validated = dict(dit_config)

    signature = inspect.signature(WanVideoDiT.__init__)
    allowed_keys = set()
    required_keys = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        allowed_keys.add(name)
        if param.default is inspect.Signature.empty:
            required_keys.add(name)

    unknown_keys = sorted(set(validated) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in `dit_config`: {unknown_keys}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )

    missing_keys = sorted(required_keys - set(validated))
    if missing_keys:
        raise ValueError(
            f"Missing required keys in `dit_config`: {missing_keys}. "
            "Please specify all required WanVideoDiT constructor args."
        )

    return validated


def _load_registered_model(
    path,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs_override: dict[str, Any] | None = None,
):
    model_hash = hash_model_file(path)

    matched_config = None
    for config in WAN22_MODEL_REGISTRY:
        if config["model_hash"] == model_hash and config["model_name"] == model_name:
            matched_config = config
            break
    if matched_config is None:
        raise ValueError(
            f"Cannot detect model type for {model_name}. File: {path}. "
            f"Model hash: {model_hash}. This standalone package follows DiffSynth hash-based loading."
        )

    model_class = matched_config["model_class"]
    model_kwargs = dict(matched_config.get("extra_kwargs", {}))
    if model_kwargs_override is not None:
        model_kwargs.update(model_kwargs_override)
    state_dict_converter = matched_config.get("state_dict_converter")

    model = model_class(**model_kwargs)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)

    keys= model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {model_name} from {path} with missing/unexpected keys: {keys}")
    model = model.to(device=device, dtype=torch_dtype)
    return model


def _resolve_configs(model_id: str, tokenizer_model_id: str, redirect_common_files: bool = True):
    dit_config = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    text_config = ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
    vae_config = ModelConfig(model_id=model_id, origin_file_pattern="Wan2.2_VAE.pth")
    tokenizer_config = ModelConfig(model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/")

    if redirect_common_files:
        redirect_dict = {
            "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
            "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
        }
        text_config.model_id, text_config.origin_file_pattern = redirect_dict[text_config.origin_file_pattern]
        vae_config.model_id, vae_config.origin_file_pattern = redirect_dict[vae_config.origin_file_pattern]
    return dit_config, text_config, vae_config, tokenizer_config


def load_wan22_ti2v_5b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
):
    logger.info("Loading Wan2.2-TI2V-5B components...")
    start = time.time()

    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.2-TI2V-5B loading.")
    validated_dit_config = _validate_dit_config(dit_config)

    dit_model_config, text_config, vae_config, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )

    vae_config.download_if_necessary()
    if load_text_encoder:
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()

    if skip_dit_load_from_pretrain:
        logger.info(
            "Skipping pretrained video DiT load (`skip_dit_load_from_pretrain=True`); "
            "initializing video expert randomly and expecting checkpoint override."
        )
        dit: WanVideoDiT = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        dit_model_config.download_if_necessary()
        dit = _load_registered_model(
            dit_model_config.path,
            "wan_video_dit",
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs_override=validated_dit_config,
        )
        dit_path = str(dit_model_config.path)
    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch_dtype,
            device=device,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); "
            "training must provide cached `context/context_mask`."
        )
    vae: WanVideoVAE38 = _load_registered_model(vae_config.path, "wan_video_vae", torch_dtype=torch_dtype, device=device)
    logger.info("Finished loading Wan2.2-TI2V-5B components in %.2f seconds.", time.time() - start)
    return Wan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        vae_path=str(vae_config.path),
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )


def _count_compatible_state_dict_keys(
    candidate_state_dict: dict[str, Any],
    target_state_dict: dict[str, torch.Tensor],
) -> int:
    return sum(
        1
        for key, value in candidate_state_dict.items()
        if key in target_state_dict
        and isinstance(value, torch.Tensor)
        and tuple(value.shape) == tuple(target_state_dict[key].shape)
    )


def _count_compatible_state_dict_elements(
    candidate_state_dict: dict[str, Any],
    target_state_dict: dict[str, torch.Tensor],
) -> int:
    return sum(
        int(value.numel())
        for key, value in candidate_state_dict.items()
        if key in target_state_dict
        and isinstance(value, torch.Tensor)
        and tuple(value.shape) == tuple(target_state_dict[key].shape)
    )


def _load_explicit_model(
    path,
    model_class,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs: dict[str, Any] | None = None,
    state_dict_converters: list[tuple[str, Any]] | None = None,
    min_key_coverage: float = 0.9,
):
    """Load official or Diffusers-format weights without relying on a fixed file hash."""
    model = model_class(**({} if model_kwargs is None else dict(model_kwargs)))
    raw_state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    variants: list[tuple[str, dict[str, Any]]] = [("raw", raw_state_dict)]
    for name, converter in state_dict_converters or []:
        try:
            variants.append((name, converter(raw_state_dict)))
        except Exception as exc:
            logger.warning("State-dict converter `%s` failed for %s: %s", name, model_class.__name__, exc)

    target_state = model.state_dict()
    name, state, key_score, element_score = max(
        (
            (
                name,
                state,
                _count_compatible_state_dict_keys(state, target_state),
                _count_compatible_state_dict_elements(state, target_state),
            )
            for name, state in variants
            if isinstance(state, dict)
        ),
        key=lambda item: (item[3], item[2]),
    )
    target_tensor_keys = sum(isinstance(value, torch.Tensor) for value in target_state.values())
    target_tensor_elements = sum(
        int(value.numel()) for value in target_state.values() if isinstance(value, torch.Tensor)
    )
    key_coverage = key_score / max(target_tensor_keys, 1)
    element_coverage = element_score / max(target_tensor_elements, 1)
    if (
        key_score <= 0
        or key_coverage < float(min_key_coverage)
        or element_coverage < float(min_key_coverage)
    ):
        raise ValueError(
            f"Insufficient compatible state-dict coverage for {model_class.__name__} in {path}: "
            f"keys={key_score}/{target_tensor_keys} ({key_coverage:.1%}), "
            f"elements={element_score}/{target_tensor_elements} ({element_coverage:.1%}), "
            f"required >= {min_key_coverage:.1%}. "
            "Refusing to continue with a mostly random backbone."
        )
    result = model.load_state_dict(state, strict=False)
    logger.info(
        "Loaded %s from %s via `%s` (keys=%d/%d, elements=%.1f%%, missing=%d, unexpected=%d).",
        model_class.__name__,
        path,
        name,
        key_score,
        target_tensor_keys,
        element_coverage * 100.0,
        len(result.missing_keys),
        len(result.unexpected_keys),
    )
    return model.to(device=device, dtype=torch_dtype)


def _resolve_configs_for_backbone(
    model_id: str,
    tokenizer_model_id: str,
    video_backbone_type: str,
    redirect_common_files: bool,
):
    spec = WAN_VIDEO_BACKBONE_SPECS[resolve_video_backbone_type(video_backbone_type)]
    dit_config = ModelConfig(model_id=model_id, origin_file_pattern=spec.dit_origin_file_pattern)
    text_config = ModelConfig(model_id=model_id, origin_file_pattern=spec.text_origin_file_pattern)
    vae_config = ModelConfig(model_id=model_id, origin_file_pattern=spec.vae_origin_file_pattern)
    tokenizer_config = ModelConfig(model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/")
    if redirect_common_files:
        text_config.model_id = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
        text_config.origin_file_pattern = "models_t5_umt5-xxl-enc-bf16.safetensors"
        if spec.backbone_type == "wan2_2_ti2v":
            vae_config.model_id = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
            vae_config.origin_file_pattern = "Wan2.2_VAE.safetensors"
    return dit_config, text_config, vae_config, tokenizer_config


def load_wan21_t2v_1_3b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = False,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
) -> Wan22LoadedComponents:
    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.1-T2V-1.3B loading.")
    validated_dit_config = _validate_dit_config(
        apply_video_backbone_preset(dit_config, "wan2_1_t2v")
    )
    dit_model_config, text_config, vae_config, tokenizer_config = _resolve_configs_for_backbone(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        video_backbone_type="wan2_1_t2v",
        redirect_common_files=redirect_common_files,
    )
    vae_config.download_if_necessary()
    if load_text_encoder:
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()

    if skip_dit_load_from_pretrain:
        dit = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        dit_model_config.download_if_necessary()
        dit = _load_explicit_model(
            dit_model_config.path,
            WanVideoDiT,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs=validated_dit_config,
            state_dict_converters=[
                ("wan_video_dit_state_dict_converter", wan_video_dit_state_dict_converter),
                ("wan_video_dit_from_diffusers", wan_video_dit_from_diffusers),
            ],
        )
        dit_path = str(dit_model_config.path)

    text_encoder = None
    tokenizer = None
    text_encoder_path = None
    tokenizer_path = None
    if load_text_encoder:
        text_encoder = _load_explicit_model(
            text_config.path,
            WanTextEncoder,
            torch_dtype=torch_dtype,
            device=device,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)

    vae = _load_explicit_model(
        vae_config.path,
        WanVideoVAE,
        torch_dtype=torch_dtype,
        device=device,
        state_dict_converters=[("wan_video_vae_state_dict_converter", wan_video_vae_state_dict_converter)],
    )
    return Wan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        vae_path=str(vae_config.path),
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )


def load_wan_video_components(
    *,
    video_backbone_type: str,
    video_backbone_name: str | None,
    model_id: str,
    tokenizer_model_id: str,
    device: str,
    torch_dtype: torch.dtype,
    tokenizer_max_len: int,
    redirect_common_files: bool,
    dit_config: dict[str, Any],
    skip_dit_load_from_pretrain: bool,
    load_text_encoder: bool,
) -> Wan22LoadedComponents:
    resolved_type = resolve_video_backbone_type(video_backbone_type)
    resolved_model_id = resolve_video_backbone_name(
        resolved_type, video_backbone_name, model_id
    )
    kwargs = dict(
        device=device,
        torch_dtype=torch_dtype,
        model_id=resolved_model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=tokenizer_max_len,
        redirect_common_files=redirect_common_files,
        dit_config=dit_config,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        load_text_encoder=load_text_encoder,
    )
    if resolved_type == "wan2_1_t2v":
        return load_wan21_t2v_1_3b_components(**kwargs)
    return load_wan22_ti2v_5b_components(**kwargs)

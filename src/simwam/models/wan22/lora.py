"""In-place LoRA utilities for the Wan video backbone.

Wraps selected ``nn.Linear`` layers with a low-rank adapter while preserving
module paths such as ``block.self_attn.q`` and base parameter identities.

Adapter: ``y = base(x) + scaling * (x · Aᵀ) · Bᵀ`` with ``scaling = alpha / r``.
``B`` is zero-initialized so the wrapped layer matches the base layer at start.

`merged_state_dict` collapses the adapters back into plain ``Linear`` weights so a LoRA
checkpoint can load into the corresponding vanilla module with no LoRA code.
"""

from __future__ import annotations

import math
from typing import Iterable, Set

import torch
import torch.nn as nn


LIGHTWAM_VIDEO_BLOCK_LORA_TARGETS = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank `r` must be > 0, got {r}")
        self.base = base
        self.base.requires_grad_(False)
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty((r, base.in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, r), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return out + self.scaling * lora

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        return (self.base.weight.data.float() + delta).to(self.base.weight.dtype)


def apply_lora_to_module(
    module: nn.Module,
    target_names: Iterable[str] = ("q", "k", "v", "o"),
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
) -> int:
    """Replace, in place, every ``nn.Linear`` child whose attribute name is in `target_names`
    with a `LoRALinear`. Recurses into submodules. Returns the number of layers wrapped."""
    targets: Set[str] = set(target_names)
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
        else:
            replaced += apply_lora_to_module(child, targets, r=r, alpha=alpha, dropout=dropout)
    return replaced


def apply_lora_to_paths(
    module: nn.Module,
    target_paths: Iterable[str],
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
) -> int:
    """Wrap the exact linear layers identified by paths relative to ``module``.

    Unlike :func:`apply_lora_to_module`, this function matches complete paths such as
    ``self_attn.q`` and ``ffn.0``.  Exact matching prevents a broad local name like
    ``0`` from accidentally adapting unrelated ``Sequential`` modules.
    """

    normalized_paths = []
    seen = set()
    for raw_path in target_paths:
        path = str(raw_path).strip()
        if not path or any(not component for component in path.split(".")):
            raise ValueError(f"Invalid LoRA target path: {raw_path!r}")
        if path in seen:
            raise ValueError(f"Duplicate LoRA target path: {path}")
        normalized_paths.append(path)
        seen.add(path)
    if not normalized_paths:
        raise ValueError("LoRA target paths must not be empty.")

    for path in normalized_paths:
        components = path.split(".")
        parent = module
        for component in components[:-1]:
            child = parent._modules.get(component)
            if child is None:
                raise ValueError(
                    f"LoRA target path `{path}` does not exist: missing `{component}`."
                )
            parent = child

        leaf = components[-1]
        child = parent._modules.get(leaf)
        if child is None:
            raise ValueError(f"LoRA target path `{path}` does not exist.")
        if isinstance(child, LoRALinear):
            raise ValueError(f"LoRA target path `{path}` is already adapted.")
        if not isinstance(child, nn.Linear):
            raise TypeError(
                f"LoRA target path `{path}` must resolve to nn.Linear, got {type(child)}."
            )
        parent._modules[leaf] = LoRALinear(
            child,
            r=int(r),
            alpha=float(alpha),
            dropout=float(dropout),
        )

    return len(normalized_paths)


def merged_state_dict(module: nn.Module) -> dict:
    """Return `module.state_dict()` with all `LoRALinear` collapsed to plain Linear weights.

    `<path>.base.weight` -> `<path>.weight` (merged), `<path>.base.bias` -> `<path>.bias`,
    and `<path>.lora_A` / `<path>.lora_B` dropped. Keys then match a vanilla module.
    """
    merged = {
        name: lin.merged_weight()
        for name, lin in module.named_modules()
        if isinstance(lin, LoRALinear)
    }
    if not merged:
        return module.state_dict()

    out = {}
    for key, value in module.state_dict().items():
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            continue
        if key.endswith(".base.weight"):
            path = key[: -len(".base.weight")]
            if path in merged:
                out[f"{path}.weight"] = merged[path]
                continue
        if key.endswith(".base.bias"):
            path = key[: -len(".base.bias")]
            if path in merged:
                out[f"{path}.bias"] = value
                continue
        out[key] = value
    return out


def remap_vanilla_to_lora_state_dict(module: nn.Module, state_dict: dict) -> dict:
    """Inverse of `merged_state_dict` (keys only): for each `LoRALinear` path in `module`,
    rename incoming vanilla `<path>.weight`/`<path>.bias` -> `<path>.base.weight`/`<path>.base.bias`
    so a merged/vanilla checkpoint loads into a LoRA-wrapped module (adapters keep their init).
    Keys not under a LoRALinear path are returned unchanged.
    """
    lora_paths = {name for name, m in module.named_modules() if isinstance(m, LoRALinear)}
    if not lora_paths:
        return state_dict
    out = {}
    for key, value in state_dict.items():
        new_key = key
        for path in lora_paths:
            if key == f"{path}.weight":
                new_key = f"{path}.base.weight"
                break
            if key == f"{path}.bias":
                new_key = f"{path}.base.bias"
                break
        out[new_key] = value
    return out

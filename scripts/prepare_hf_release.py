"""Prepare the reported epoch-100 adaptation checkpoint for Hugging Face.

Uses CPU only. Does not upload files or read the training run configuration.
"""

import argparse
import hashlib
import json
import pickletools
import re
import shutil
import zipfile
from pathlib import Path

import torch


EXPECTED_SHA256 = "7b880759c6c4244cb9123cfd9320eed303dd9b6204979733fbfe4c96b95c52c5"
STATE_FIELDS = ("video_expert", "adapters", "trajectory_head", "proprio_encoder")
EXPECTED_LORA = {
    "enabled": True,
    "layer_indices": list(range(30)),
    "target_modules": [
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        "ffn.0", "ffn.2",
    ],
    "rank": 64,
    "alpha": 128.0,
    "dropout": 0.0,
}
PRIVATE_TEXT = re.compile(
    r"/(?:Users|home|inspire|mnt|scratch)/|[A-Z]:\\Users\\|global_user|"
    r"job-[0-9a-f-]{20,}|"
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}|"
    r"(?:hf_|gh[pousr]_)[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"sk-[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{16}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    re.IGNORECASE,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_text(value):
    if PRIVATE_TEXT.search(value):
        # Do not echo the potentially private value into terminal logs.
        raise ValueError("Potential private path, identifier or credential; release aborted.")


def check_value(value):
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, str):
        check_text(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Unexpected checkpoint dictionary key type.")
            check_text(key)
            check_value(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_value(item)
    elif value is not None and type(value) not in (bool, int, float):
        raise ValueError("Unexpected non-tensor object in checkpoint.")


def inspect_checkpoint(path):
    # Inspect serialized strings too, including state-dict metadata attributes.
    with zipfile.ZipFile(path) as archive:
        pickles = []
        for name in archive.namelist():
            check_text(name)
            if name.endswith("/data.pkl") or name == "data.pkl":
                pickles.append(name)
        if len(pickles) != 1:
            raise ValueError("Expected one PyTorch checkpoint metadata stream.")
        for _, argument, _ in pickletools.genops(archive.read(pickles[0])):
            if isinstance(argument, str):
                check_text(argument)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = set(STATE_FIELDS) | {
        "format", "step", "video_expert_is_peft", "tap_layers", "backbone_lora",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Unexpected checkpoint fields; optimizer/run metadata is not allowed.")
    if (payload["format"] != "lightwam_ad_v1" or payload["step"] != 131700
            or payload["video_expert_is_peft"] is not True
            or payload["tap_layers"] != [8, 16, 24]
            or payload["backbone_lora"] != EXPECTED_LORA):
        raise ValueError("Checkpoint does not match the reported epoch-100 configuration.")
    for field in STATE_FIELDS:
        state = payload[field]
        if not isinstance(state, dict) or not state:
            raise ValueError("Missing checkpoint tensor state.")
        if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError("Non-tensor content found in a module state dictionary.")
    if not all(k.startswith("head.") or k.endswith(("lora_A", "lora_B"))
               for k in payload["video_expert"]):
        raise ValueError("Expected adaptation weights, without the frozen backbone.")
    check_value(payload)


def prepare_release(checkpoint, output_dir, repo_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
        raise ValueError("Use a Hugging Face repository ID: username/model-name.")
    checkpoint, output_dir = Path(checkpoint), Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Use a new or empty release directory.")
    if sha256_file(checkpoint) != EXPECTED_SHA256:
        raise ValueError("SHA256 differs from the evaluated checkpoint; release aborted.")
    inspect_checkpoint(checkpoint)

    template = Path(__file__).resolve().parents[1] / "docs" / "huggingface_model_card.md"
    card = template.read_text().replace("{{REPO_ID}}", repo_id)
    check_text(card)
    metadata = {
        "format": "lightwam_ad_v1", "epoch": 100, "step": 131700,
        "checkpoint": "step_131700.pt", "sha256": EXPECTED_SHA256,
        "base_model": "Wan-AI/Wan2.1-T2V-1.3B",
        "task": "lightwam_ad_navsim_front_384x672",
        "tap_layers": [8, 16, 24], "backbone_lora": EXPECTED_LORA,
        "training": {"epochs": 100, "effective_batch_size": 64, "precision": "bf16",
                     "learning_rate": 1e-4, "optimizer": "AdamW", "weight_decay": 0.01,
                     "lr_scheduler": "cosine", "seed": 42,
                     "train_clips": 84258, "validation_clips": 851},
        "navsim_v2_epdms_100": {"navtest": 88.576, "navhard": 33.281},
        "code": "https://github.com/L1ziang/LightWAM-AD/tree/lightwam-ad",
    }
    check_value(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "step_131700.pt"
    shutil.copyfile(checkpoint, target)
    if sha256_file(target) != EXPECTED_SHA256:
        raise ValueError("Copied checkpoint failed SHA256 verification.")
    (output_dir / "README.md").write_text(card)
    (output_dir / "release.json").write_text(json.dumps(metadata, indent=2) + "\n")
    names = ("step_131700.pt", "README.md", "release.json")
    (output_dir / "SHA256SUMS").write_text("".join(
        f"{sha256_file(output_dir / name)}  {name}\n" for name in names
    ))
    print("Checkpoint hash, format and metadata checks passed.")
    for name in (*names, "SHA256SUMS"):
        print(f"{name}: {(output_dir / name).stat().st_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    prepare_release(args.checkpoint, args.output_dir, args.repo_id)

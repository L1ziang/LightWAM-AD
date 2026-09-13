import logging
import json
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_final_checkpoint = bool(cfg.get("save_final_checkpoint", True))
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        eval_num_batches = cfg.get("eval_num_batches")
        self.eval_num_batches = None if eval_num_batches is None else int(eval_num_batches)
        if self.eval_num_batches is not None and self.eval_num_batches <= 0:
            raise ValueError(f"`eval_num_batches` must be positive or null, got {eval_num_batches}.")
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with="tensorboard",
            project_dir=self.output_dir,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Apply the architecture-specific freeze policy before ZeRO constructs
        # optimizer state. Legacy SimWAM falls back to DiT/proprio-only training.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_params:
            raise RuntimeError("Model freeze policy left no trainable parameters.")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        ## build dataloader with accelerator.prepare_sampler for proper distributed sampling under ZeRO2
        self.train_loader = self._build_train_loader(
            self.train_dataset, worker_init_fn=worker_init_fn
        )
        self.val_loader = self._build_val_loader(
            self.val_dataset, worker_init_fn=worker_init_fn
        )
        
        ## calculate total training steps
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        
        ## get lr scheduler
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self._init_tensorboard()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else 0
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)
        self._log_run_metadata()

    def _init_tensorboard(self):
        self.accelerator.init_trackers("train")
        logger.info("Initialized TensorBoard tracker under output_dir=%s", self.output_dir)

    def _log_run_metadata(self):
        if not self.accelerator.is_main_process:
            return
        effective_global_batch = (
            self.batch_size
            * self.accelerator.num_processes
            * self.gradient_accumulation_steps
        )
        metadata = {
            "data/train_samples": float(len(self.train_dataset)),
            "data/val_samples": float(len(self.val_dataset) if self.val_dataset is not None else 0),
            "data/effective_global_batch_size": float(effective_global_batch),
            "run/world_size": float(self.accelerator.num_processes),
            "run/max_steps": float(self.max_steps),
        }
        split_metadata = getattr(self.train_dataset, "split_metadata", None)
        if split_metadata is not None:
            metadata["data/val_fraction"] = float(split_metadata.val_fraction)
            metadata["data/split_seed"] = float(split_metadata.seed)
            split_manifest = {
                **split_metadata.to_dict(),
                "validation_tokens": list(getattr(self.val_dataset, "tokens", [])),
            }
            split_manifest_path = Path(self.output_dir) / "data_split.json"
            with open(split_manifest_path, "w", encoding="utf-8") as file:
                json.dump(split_manifest, file, ensure_ascii=True, indent=2, sort_keys=True)
            logger.info("Wrote deterministic data split manifest: %s", split_manifest_path)
        self._tensorboard_log(metadata)

        try:
            writer = self.accelerator.get_tracker("tensorboard", unwrap=True)
            config_text = json.dumps(
                {
                    "output_dir": self.output_dir,
                    "train_samples": len(self.train_dataset),
                    "val_samples": len(self.val_dataset) if self.val_dataset is not None else 0,
                    "effective_global_batch_size": effective_global_batch,
                    "max_steps": self.max_steps,
                    "split": None if split_metadata is None else split_metadata.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
            writer.add_text("run/metadata", f"```json\n{config_text}\n```", self.global_step)
        except Exception as exc:
            logger.warning("Unable to add TensorBoard text metadata: %s", exc)

    def _tensorboard_log(self, payload: dict):
        self.accelerator.log(payload, step=self.global_step)

    def _finish_tensorboard(self):
        self.accelerator.end_training()

    def _build_train_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _build_val_loader(self, dataset, worker_init_fn=None):
        if dataset is None:
            return None
        # Do not pass this loader through Accelerator.prepare(): each rank receives an
        # explicit, non-padded strided subset.  This evaluates every validation sample
        # exactly once even when len(val) is not divisible by world size.
        rank_indices = list(
            range(
                self.accelerator.process_index,
                len(dataset),
                self.accelerator.num_processes,
            )
        )
        rank_dataset = Subset(dataset, rank_indices)
        return DataLoader(
            rank_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        logger.info("Applying the model-specific trainable-module policy.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        if hasattr(model, "configure_trainable_modules"):
            model.configure_trainable_modules()
            return
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            
            # TODO and NOTE: since the VAE encoder use the first frame as reference, and the future frames must be 4 times;
            # if using history 4 frame as reference in autonomous trajectory prediction, the action horizon must be divisible by 
            #  (num_frames - 4 - 1) which is the number of video transitions after the first 4 history frames.
            
            # if action_dim == 3 or action_dim == 2: ## for autonomous traj
            #     if action_horizon % (num_frames - 4 - 1) != 0 or action_horizon % (num_frames - 1) != 0:  ## the first 4 frame are reference frame
            #         raise ValueError(
            #             f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 4 - 1}), got {action_horizon}"
            #         )
            # else:
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,     # [1, 3, T, H, W]
            "prompt": prompt,   
            "action": action,   # [1, Ta, Da]
            "proprio": proprio, # [1, T, Dp]
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def _evaluate_validation_loss(self, model) -> dict[str, float]:
        if self.val_loader is None:
            return {}

        local_sums: dict[str, float] = {"val_loss": 0.0}
        local_count = 0
        cuda_devices = []
        if self.accelerator.device.type == "cuda":
            cuda_devices = [self.accelerator.device.index]

        # training_loss samples diffusion noise/timesteps.  Fork and restore RNG state so
        # changing eval frequency cannot change the subsequent optimization trajectory.
        with torch.random.fork_rng(devices=cuda_devices):
            eval_seed = self.seed + self.global_step * 1009 + self.accelerator.process_index
            torch.manual_seed(eval_seed)
            if self.accelerator.device.type == "cuda":
                torch.cuda.manual_seed(eval_seed)

            for batch_index, sample in enumerate(self.val_loader):
                if self.eval_num_batches is not None and batch_index >= self.eval_num_batches:
                    break
                if "video" not in sample or not isinstance(sample["video"], torch.Tensor):
                    raise TypeError("Validation samples must contain a batched tensor `video`.")
                batch_size = int(sample["video"].shape[0])
                with self.accelerator.autocast():
                    val_loss, loss_dict = model.training_loss(sample)
                local_sums["val_loss"] += float(val_loss.detach().float().item()) * batch_size
                for key, value in loss_dict.items():
                    local_sums.setdefault(str(key), 0.0)
                    local_sums[str(key)] += float(value) * batch_size
                local_count += batch_size

        metric_names = sorted(local_sums)
        packed = torch.tensor(
            [local_sums[name] for name in metric_names] + [float(local_count)],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        packed = self.accelerator.reduce(packed, reduction="sum")
        global_count = int(round(float(packed[-1].item())))
        if global_count <= 0:
            raise RuntimeError("Validation loader yielded zero samples across all ranks.")
        metrics = {
            name: float(packed[index].item() / global_count)
            for index, name in enumerate(metric_names)
        }
        metrics["num_samples"] = float(global_count)
        return metrics

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        validation_metrics = self._evaluate_validation_loss(model)

        # Keep one deterministic inference sample per rank for trajectory/video diagnostics;
        # the validation losses above are aggregated over the complete configured holdout.
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )

        if pred.get("video") is None:
            pred_action = pred.get("action")
            action_l1 = None
            action_l2 = None
            if action is not None and pred_action is not None:
                pred_btd = pred_action.unsqueeze(0) if pred_action.ndim == 2 else pred_action
                gt_btd = action.unsqueeze(0) if action.ndim == 2 else action
                pred_denorm = self.val_dataset.denormalize_action(pred_btd)
                gt_denorm = self.val_dataset.denormalize_action(gt_btd)
                if pred_denorm.shape != gt_denorm.shape:
                    raise ValueError(
                        "Direct action prediction/target shape mismatch: "
                        f"{tuple(pred_denorm.shape)} vs {tuple(gt_denorm.shape)}"
                    )
                diff = pred_denorm - gt_denorm
                action_l1 = diff.abs().mean().item()
                action_l2 = diff.pow(2).mean().item()
            local_metrics = torch.tensor(
                [
                    -1.0 if action_l2 is None else float(action_l2),
                    -1.0 if action_l1 is None else float(action_l1),
                ],
                device=self.accelerator.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            gathered = self.accelerator.gather_for_metrics(local_metrics)
            means = gathered.mean(dim=0)
            if was_dit_training or hasattr(model, "configure_trainable_modules"):
                self._set_dit_only_train_mode()
            result = dict(validation_metrics)
            if action_l2 is not None:
                result["action_l2"] = float(means[0].item())
                result["action_l1"] = float(means[1].item())
            return result
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            denorm_actions = {}
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                denorm_action = self.val_dataset.denormalize_action(action_btd)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :6].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 6].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 7].mean().item() if action_l1 is not None else None

        if was_dit_training or hasattr(model, "configure_trainable_modules"):
            self._set_dit_only_train_mode()

        result = {
            **validation_metrics,
            "psnr_rg": float(mean_metrics[0].item()),
            "ssim_rg": float(mean_metrics[1].item()),
            "psnr_rd": float(mean_metrics[2].item()),
            "ssim_rd": float(mean_metrics[3].item()),
            "psnr_dg": float(mean_metrics[4].item()),
            "ssim_dg": float(mean_metrics[5].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        # unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        try:
            while self.global_step < self.max_steps:
                try:
                    sample = next(data_iter)
                    self.batch_in_epoch += 1
                except StopIteration:
                    self.epoch += 1
                    self.batch_in_epoch = 0
                    self.train_sampler.clear_resume_batch_offset()
                    self.train_sampler.set_epoch_offset(0)
                    self.train_sampler.set_epoch(self.epoch)
                    data_iter = iter(self.train_loader)
                    continue

                with self.accelerator.accumulate(self.model):
                    ### mixed_precision training step
                    with self.accelerator.autocast():
                        # Invoke the prepared wrapper's forward method so DDP/DeepSpeed
                        # forward hooks and synchronization are not bypassed.
                        loss, loss_dict = self.model(sample)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        if not self.accelerator.optimizer_step_was_skipped:
                            self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1
                        global_loss = float(
                            self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                        )
                        global_loss_metrics = {}
                        for key, value in loss_dict.items():
                            metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                            global_loss_metrics[key] = float(
                                self.accelerator.gather(metric_tensor).mean().item()
                            )
                        grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                        global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                        current_lr = float(self.optimizer.param_groups[0]["lr"])

                        if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                            eta_str, steps_per_sec = self._estimate_eta()
                            samples_per_sec = (
                                steps_per_sec
                                * self.batch_size
                                * self.accelerator.num_processes
                                * self.gradient_accumulation_steps
                            )
                            description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                                self.epoch,
                                self.global_step,
                                self.max_steps,
                                global_loss,
                            )
                            if global_loss_metrics:
                                detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                                description += detail_str + " "
                            description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                                current_lr,
                                steps_per_sec,
                                samples_per_sec,
                                eta_str,
                            )
                            logger.info(description)

                            tensorboard_payload = {
                                "train/loss": global_loss,
                                "train/epoch": float(self.epoch),
                                "train/batch_in_epoch": float(self.batch_in_epoch),
                                "train/grad_norm": global_grad_norm,
                                "train/lr": current_lr,
                                "performance/steps_per_sec": steps_per_sec,
                                "performance/samples_per_sec": samples_per_sec,
                            }
                            if self.accelerator.device.type == "cuda":
                                tensorboard_payload["memory/gpu_allocated_gib"] = float(
                                    torch.cuda.memory_allocated(self.accelerator.device) / 1024**3
                                )
                                tensorboard_payload["memory/gpu_reserved_gib"] = float(
                                    torch.cuda.memory_reserved(self.accelerator.device) / 1024**3
                                )
                            for key, value in global_loss_metrics.items():
                                tensorboard_payload[f"train/{key}"] = value
                            self._tensorboard_log(tensorboard_payload)

                        if (
                            self.eval_every > 0
                            and self.val_dataset is not None
                            and self.global_step % self.eval_every == 0
                        ):
                            metrics = self.evaluate()
                            self.accelerator.wait_for_everyone()
                            if metrics is not None and self.accelerator.is_main_process:
                                description = "[eval] step=%d val_loss=%.4f" % (
                                    self.global_step, metrics["val_loss"]
                                )
                                if "psnr_rd" in metrics:
                                    description += " infer_psnr=%.4f infer_ssim=%.4f" % (
                                        metrics["psnr_rd"], metrics["ssim_rd"]
                                    )
                                if "action_l2" in metrics:
                                    description += " action_l2=%.4f" % metrics["action_l2"]
                                if "action_l1" in metrics:
                                    description += " action_l1=%.4f" % metrics["action_l1"]
                                description += " samples=%d" % int(metrics.get("num_samples", 0))
                                logger.info(description)
                                eval_payload = {
                                    "eval/loss": float(metrics["val_loss"]),
                                    "eval/num_samples": float(metrics.get("num_samples", 0)),
                                }
                                for loss_name in (
                                    "loss_video", "loss_action", "loss_video_raw", "loss_action_raw"
                                ):
                                    if loss_name in metrics:
                                        eval_payload[f"eval/{loss_name}"] = float(metrics[loss_name])
                                for metric_name in (
                                    "psnr_rg", "ssim_rg", "psnr_rd", "ssim_rd", "psnr_dg", "ssim_dg"
                                ):
                                    if metric_name in metrics:
                                        eval_payload[f"eval/{metric_name}"] = float(metrics[metric_name])
                                if "action_l2" in metrics:
                                    eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                                if "action_l1" in metrics:
                                    eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                                self._tensorboard_log(eval_payload)

                        if self.save_every > 0 and self.global_step % self.save_every == 0:
                            ckpt_info = self.save_checkpoint()
                            if self.accelerator.is_main_process:
                                logger.info(
                                    "[ckpt] step=%d weights=%s state=%s",
                                    self.global_step,
                                    ckpt_info["weights_path"],
                                    ckpt_info["state_path"],
                                )

                        if self.global_step >= self.max_steps:
                            if self.save_final_checkpoint:
                                ckpt_info = self.save_checkpoint()
                                if self.accelerator.is_main_process:
                                    logger.info(
                                        "[done] max_steps reached step=%d weights=%s state=%s",
                                        self.global_step,
                                        ckpt_info["weights_path"],
                                        ckpt_info["state_path"],
                                    )
                            elif self.accelerator.is_main_process:
                                logger.info("[done] max_steps reached step=%d (final checkpoint disabled)", self.global_step)
                            return

            if self.save_final_checkpoint:
                ckpt_info = self.save_checkpoint()
                if self.accelerator.is_main_process:
                    logger.info(
                        "[done] training finished step=%d weights=%s state=%s",
                        self.global_step,
                        ckpt_info["weights_path"],
                        ckpt_info["state_path"],
                    )
        finally:
            self._finish_tensorboard()
        

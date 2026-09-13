"""Small, testable helpers for serial CUDA inference benchmarks."""
from __future__ import annotations

import math
import random
from contextlib import contextmanager
from functools import wraps
from typing import Callable

import numpy as np
import torch


def summarize_ms(values) -> dict:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("Expected nonempty, finite, nonnegative timing samples.")
    return {
        "count": len(data),
        "mean_ms": float(data.mean()),
        "std_ms": float(data.std()),
        "min_ms": float(data.min()),
        "p50_ms": float(np.percentile(data, 50)),
        "p95_ms": float(np.percentile(data, 95)),
        "p99_ms": float(np.percentile(data, 99)),
        "max_ms": float(data.max()),
    }


def select_tokens(tokens, count: int, seed: int) -> list[str]:
    ordered = sorted(str(token) for token in tokens)
    if len(set(ordered)) != len(ordered):
        raise ValueError("Scene loader contains duplicate tokens.")
    if count <= 0 or count > len(ordered):
        raise ValueError(f"Requested {count} distinct scenes; available={len(ordered)}.")
    return random.Random(seed).sample(ordered, count)


def compare_actions(reference: torch.Tensor, candidate: torch.Tensor, atol: float) -> dict:
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("Parity tolerance must be finite and nonnegative.")
    if reference.shape != candidate.shape:
        raise ValueError("Parity output shapes differ.")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("Nonfinite trajectory in parity check.")
    error = float((reference.double() - candidate.double()).abs().max().item())
    return {"exact": torch.equal(reference, candidate), "max_abs_error": error, "passed": error <= atol}


def parameter_counts(model: torch.nn.Module) -> dict:
    """Count unique registered parameters; preserve original training flags."""
    return {
        "total_loaded": sum(p.numel() for p in model.parameters()),
        "trainable_by_config": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
        "by_top_level_module": {
            name: sum(p.numel() for p in module.parameters())
            for name, module in model.named_children()
        },
    }


class CudaStageRecorder:
    """CUDA event regions for a *separate* profiling pass, not headline latency.

    Nested head/resampler/decoder spans overlap and must not be added together.
    Elapsed event time includes device-timeline gaps, not just summed kernel time.
    """

    def __init__(self, event_factory: Callable | None = None):
        self.event_factory = event_factory or (lambda: torch.cuda.Event(enable_timing=True))
        self.events = []

    def clear(self):
        self.events.clear()

    def wrap(self, name, original):
        @wraps(original)
        def measured(*args, **kwargs):
            start, end = self.event_factory(), self.event_factory()
            start.record()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                self.events.append((name, start, end))
        return measured

    @contextmanager
    def instrument(self, targets):
        """Restore descriptors/instance overrides even when inference raises."""
        sentinel = object()
        originals = []
        try:
            for name, obj, attribute in targets:
                previous = vars(obj).get(attribute, sentinel)
                original = getattr(obj, attribute)
                setattr(obj, attribute, self.wrap(name, original))
                originals.append((obj, attribute, previous))
            yield self
        finally:
            for obj, attribute, previous in reversed(originals):
                if previous is sentinel:
                    delattr(obj, attribute)
                else:
                    setattr(obj, attribute, previous)

    def elapsed_ms(self) -> dict[str, float]:
        # Caller synchronizes the device before reading the events.
        result = {}
        for name, start, end in self.events:
            result[name] = result.get(name, 0.0) + float(start.elapsed_time(end))
        return result

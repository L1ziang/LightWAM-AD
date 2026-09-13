"""Deterministic dataset splitting helpers.

The validation membership is ranked by a stable hash of each sample identifier instead
of relying on PyTorch's RNG implementation or the dataset enumeration order.  This makes
the same seed reproduce the same NAVSIM token split across hosts and distributed ranks.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSplitMetadata:
    strategy: str
    seed: int
    val_fraction: float
    total_size: int
    train_size: int
    val_size: int
    val_ids_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DelegatingSubset(Dataset):
    """A subset that preserves dataset-specific methods such as action denormalization."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        *,
        split_name: str,
        split_metadata: DatasetSplitMetadata,
    ) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.split_name = str(split_name)
        self.split_metadata = split_metadata

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    @property
    def tokens(self) -> list[str]:
        tokens = getattr(self.dataset, "tokens", None)
        if tokens is None:
            raise AttributeError("The wrapped dataset does not expose `tokens`.")
        return [str(tokens[index]) for index in self.indices]

    def __getattr__(self, name: str):
        # Called only when normal attribute lookup failed.  Keep Dataset/DataLoader
        # compatibility while forwarding domain helpers to the wrapped dataset.
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)


def _sample_ids(dataset: Dataset) -> list[str]:
    tokens = getattr(dataset, "tokens", None)
    if tokens is not None:
        if len(tokens) != len(dataset):
            raise ValueError(
                "Dataset `tokens` length does not match dataset length: "
                f"{len(tokens)} != {len(dataset)}"
            )
        sample_ids = [str(token) for token in tokens]
    else:
        sample_ids = [str(index) for index in range(len(dataset))]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Stable random splitting requires unique sample identifiers.")
    return sample_ids


def deterministic_token_hash_split(
    dataset: Dataset,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[DelegatingSubset, DelegatingSubset]:
    """Split a dataset by stable pseudorandom ranking of its sample identifiers."""

    total_size = len(dataset)
    val_fraction = float(val_fraction)
    seed = int(seed)
    if total_size < 2:
        raise ValueError(f"Random train/val splitting requires at least 2 samples, got {total_size}.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"`val_fraction` must be in (0, 1), got {val_fraction}.")

    val_size = min(max(int(round(total_size * val_fraction)), 1), total_size - 1)
    sample_ids = _sample_ids(dataset)

    def rank_key(index: int) -> tuple[bytes, str]:
        sample_id = sample_ids[index]
        digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
        return digest, sample_id

    ranked_indices = sorted(range(total_size), key=rank_key)
    val_index_set = set(ranked_indices[:val_size])
    # Preserve the base dataset's order inside each subset for predictable sequential
    # validation I/O.  Training still shuffles through ResumableEpochSampler.
    train_indices = [index for index in range(total_size) if index not in val_index_set]
    val_indices = [index for index in range(total_size) if index in val_index_set]
    val_ids = sorted(sample_ids[index] for index in val_indices)
    val_ids_sha256 = hashlib.sha256("\n".join(val_ids).encode("utf-8")).hexdigest()
    metadata = DatasetSplitMetadata(
        strategy="token_hash",
        seed=seed,
        val_fraction=val_fraction,
        total_size=total_size,
        train_size=len(train_indices),
        val_size=len(val_indices),
        val_ids_sha256=val_ids_sha256,
    )
    return (
        DelegatingSubset(
            dataset,
            train_indices,
            split_name="train",
            split_metadata=metadata,
        ),
        DelegatingSubset(
            dataset,
            val_indices,
            split_name="val",
            split_metadata=metadata,
        ),
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, Tensor]


class SyntheticProxyIVDataset(Dataset):
    """
    General in-memory dataset wrapper for synthetic Proxy-IV data.
    """

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        keys: Optional[Iterable[str]] = None,
        required_keys: Optional[Sequence[str]] = None,
        default_dtype: torch.dtype = torch.float32,
        dtype_overrides: Optional[Mapping[str, torch.dtype]] = None,
    ) -> None:
        if not arrays:
            raise ValueError("`arrays` must be non-empty.")

        self.available_keys = tuple(arrays.keys())
        self.required_keys = tuple(required_keys or ())
        self.default_dtype = default_dtype
        self.dtype_overrides = dict(dtype_overrides or {})

        for key in self.required_keys:
            if key not in arrays:
                raise KeyError(f"Missing required key: {key}")

        if keys is None:
            keys = tuple(arrays.keys())
        else:
            keys = tuple(keys)

        if not keys:
            raise ValueError("`keys` must contain at least one field.")

        for key in keys:
            if key not in arrays:
                raise KeyError(f"Requested key `{key}` is not present in arrays.")

        first_key = keys[0]
        n = len(np.asarray(arrays[first_key]))

        tensor_dict: TensorDict = {}
        for key in keys:
            arr = np.asarray(arrays[key])

            if len(arr) != n:
                raise ValueError(
                    f"Inconsistent leading dimension for key `{key}`. "
                    f"Expected {n}, got {len(arr)}."
                )

            dtype = self.dtype_overrides.get(key, self.default_dtype)
            tensor_dict[key] = torch.as_tensor(arr, dtype=dtype)

        self.keys = keys
        self.tensors = tensor_dict
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> TensorDict:
        return {k: v[idx] for k, v in self.tensors.items()}

    def has_key(self, key: str) -> bool:
        return key in self.tensors

    def get_tensor(self, key: str) -> Tensor:
        if key not in self.tensors:
            raise KeyError(f"Key `{key}` is not in the active dataset view.")
        return self.tensors[key]

    def feature_dims(self) -> Dict[str, Tuple[int, ...]]:
        dims: Dict[str, Tuple[int, ...]] = {}
        for key, value in self.tensors.items():
            dims[key] = tuple(value.shape[1:])
        return dims

    def subset(self, keys: Iterable[str]) -> "SyntheticProxyIVDataset":
        new_keys = tuple(keys)
        arrays = {k: self.tensors[k].cpu().numpy() for k in new_keys}
        return SyntheticProxyIVDataset(
            arrays=arrays,
            keys=new_keys,
            required_keys=(),
            default_dtype=self.default_dtype,
            dtype_overrides=self.dtype_overrides,
        )


def split_array_dict(
    arrays: Mapping[str, np.ndarray],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 0,
    shuffle: bool = True,
) -> Tuple[ArrayDict, ArrayDict, ArrayDict]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError("`train_frac` must lie in (0, 1).")
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("`val_frac` must lie in [0, 1).")
    if train_frac + val_frac >= 1.0:
        raise ValueError("Need train_frac + val_frac < 1.0.")

    keys = tuple(arrays.keys())
    if not keys:
        raise ValueError("`arrays` must be non-empty.")

    n = len(np.asarray(arrays[keys[0]]))
    for key in keys[1:]:
        arr = np.asarray(arrays[key])
        if len(arr) != n:
            raise ValueError(f"Array `{key}` has length {len(arr)}, expected {n}.")

    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]

    def take(idx: np.ndarray) -> ArrayDict:
        return {k: np.asarray(v)[idx] for k, v in arrays.items()}

    return take(idx_train), take(idx_val), take(idx_test)


def make_synthetic_proxyiv_datasets(
    arrays: Mapping[str, np.ndarray],
    keys: Optional[Iterable[str]] = None,
    required_keys: Optional[Sequence[str]] = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 0,
    shuffle: bool = True,
    default_dtype: torch.dtype = torch.float32,
    dtype_overrides: Optional[Mapping[str, torch.dtype]] = None,
) -> Tuple[SyntheticProxyIVDataset, SyntheticProxyIVDataset, SyntheticProxyIVDataset]:
    train_arrays, val_arrays, test_arrays = split_array_dict(
        arrays=arrays,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
        shuffle=shuffle,
    )

    train_ds = SyntheticProxyIVDataset(
        arrays=train_arrays,
        keys=keys,
        required_keys=required_keys,
        default_dtype=default_dtype,
        dtype_overrides=dtype_overrides,
    )
    val_ds = SyntheticProxyIVDataset(
        arrays=val_arrays,
        keys=keys,
        required_keys=required_keys,
        default_dtype=default_dtype,
        dtype_overrides=dtype_overrides,
    )
    test_ds = SyntheticProxyIVDataset(
        arrays=test_arrays,
        keys=keys,
        required_keys=required_keys,
        default_dtype=default_dtype,
        dtype_overrides=dtype_overrides,
    )
    return train_ds, val_ds, test_ds


@dataclass
class DataLoaderBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def make_synthetic_proxyiv_dataloaders(
    arrays: Mapping[str, np.ndarray],
    keys: Optional[Iterable[str]] = None,
    required_keys: Optional[Sequence[str]] = None,
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 0,
    shuffle_before_split: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last_train: bool = False,
    default_dtype: torch.dtype = torch.float32,
    dtype_overrides: Optional[Mapping[str, torch.dtype]] = None,
) -> DataLoaderBundle:
    train_ds, val_ds, test_ds = make_synthetic_proxyiv_datasets(
        arrays=arrays,
        keys=keys,
        required_keys=required_keys,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
        shuffle=shuffle_before_split,
        default_dtype=default_dtype,
        dtype_overrides=dtype_overrides,
    )

    if eval_batch_size is None:
        eval_batch_size = batch_size

    train_gen = torch.Generator()
    train_gen.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=train_gen,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return DataLoaderBundle(train=train_loader, val=val_loader, test=test_loader)


__all__ = [
    "DataLoaderBundle",
    "SyntheticProxyIVDataset",
    "make_synthetic_proxyiv_dataloaders",
    "make_synthetic_proxyiv_datasets",
    "split_array_dict",
]
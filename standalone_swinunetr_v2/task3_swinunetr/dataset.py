from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


REGION_NAMES = ("WT", "TC", "ET")


def list_case_paths(data_dir: str | Path) -> list[Path]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")
    case_paths = sorted(data_path.glob("*.npz"))
    if not case_paths:
        raise FileNotFoundError(f"No .npz files found under: {data_path}")
    return case_paths


def split_case_paths(case_paths: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    shuffled = list(case_paths)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_paths = shuffled[:val_count]
    train_paths = shuffled[val_count:]
    if not train_paths:
        raise ValueError("Validation split left no training cases. Reduce val_ratio.")
    return train_paths, val_paths


def _compute_crop_start(center: int, crop: int, length: int) -> int:
    start = center - crop // 2
    start = max(0, start)
    start = min(start, max(0, length - crop))
    return start


def _pad_spatial_nd(array: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    spatial = array.shape[-3:]
    pad_width = [(0, 0)] * array.ndim
    for axis, (current, target) in enumerate(zip(spatial, target_shape), start=array.ndim - 3):
        delta = max(0, target - current)
        before = delta // 2
        after = delta - before
        pad_width[axis] = (before, after)
    return np.pad(array, pad_width=pad_width, mode="constant")


def _crop_or_pad_nd(array: np.ndarray, start: tuple[int, int, int], size: tuple[int, int, int]) -> np.ndarray:
    slices = tuple(slice(s, s + c) for s, c in zip(start, size))
    cropped = array[(slice(None),) * (array.ndim - 3) + slices]
    return _pad_spatial_nd(cropped, size)


def random_foreground_crop(
    image: np.ndarray,
    target: np.ndarray,
    patch_size: tuple[int, int, int],
    foreground_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    spatial_shape = image.shape[-3:]
    target_union = target.sum(axis=0) > 0

    if target_union.any() and random.random() < foreground_prob:
        fg_coords = np.argwhere(target_union)
        center = fg_coords[random.randrange(len(fg_coords))]
    else:
        center = np.array([random.randrange(max(1, dim)) for dim in spatial_shape])

    start = tuple(_compute_crop_start(int(center[i]), patch_size[i], spatial_shape[i]) for i in range(3))
    return (
        _crop_or_pad_nd(image, start, patch_size),
        _crop_or_pad_nd(target, start, patch_size),
    )


def apply_patch_augmentations(
    image: np.ndarray,
    target: np.ndarray,
    flip_prob: float,
    intensity_scale: float,
    intensity_shift: float,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    for axis in range(3):
        if random.random() < flip_prob:
            image = np.flip(image, axis=axis + 1)
            target = np.flip(target, axis=axis + 1)

    if intensity_scale > 0:
        scale = 1.0 + random.uniform(-intensity_scale, intensity_scale)
        image = image * scale

    if intensity_shift > 0:
        shift = random.uniform(-intensity_shift, intensity_shift)
        image = image + shift

    if noise_std > 0:
        sigma = random.uniform(0.0, noise_std)
        image = image + np.random.normal(loc=0.0, scale=sigma, size=image.shape).astype(np.float32)

    return image, target


class BraTS3DPatchDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        case_paths: list[Path],
        patch_size: tuple[int, int, int],
        samples_per_case: int = 2,
        foreground_prob: float = 0.8,
        augment: bool = False,
        flip_prob: float = 0.0,
        intensity_scale: float = 0.0,
        intensity_shift: float = 0.0,
        noise_std: float = 0.0,
    ) -> None:
        self.case_paths = case_paths
        self.patch_size = patch_size
        self.samples_per_case = samples_per_case
        self.foreground_prob = foreground_prob
        self.augment = augment
        self.flip_prob = flip_prob
        self.intensity_scale = intensity_scale
        self.intensity_shift = intensity_shift
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.case_paths) * self.samples_per_case

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        case_path = self.case_paths[index % len(self.case_paths)]
        data = np.load(case_path)
        image = data["image"].astype(np.float32)
        target = data["regions"].astype(np.float32)
        image, target = random_foreground_crop(image, target, self.patch_size, self.foreground_prob)
        if self.augment:
            image, target = apply_patch_augmentations(
                image=image,
                target=target,
                flip_prob=self.flip_prob,
                intensity_scale=self.intensity_scale,
                intensity_shift=self.intensity_shift,
                noise_std=self.noise_std,
            )
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "case_id": case_path.stem,
        }


class BraTS3DVolumeDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, case_paths: list[Path]) -> None:
        self.case_paths = case_paths

    def __len__(self) -> int:
        return len(self.case_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        case_path = self.case_paths[index]
        data = np.load(case_path)
        image = data["image"].astype(np.float32)
        target = data["regions"].astype(np.float32)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "case_id": case_path.stem,
        }

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from task3_swinunetr.dataset import list_case_paths

try:
    from monai.networks.nets import SwinUNETR
except ImportError as exc:
    raise SystemExit("MONAI is not installed. Please install dependencies first.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal SwinUNETR smoke test.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="outputs_task1/processed_3d",
        help="Directory containing Task 1 3D .npz files.",
    )
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--feature-size", type=int, default=24)
    parser.add_argument("--disable-v2", action="store_true")
    return parser.parse_args()


def center_crop_or_pad(image: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    channels = image.shape[0]
    output = np.zeros((channels, *target_shape), dtype=np.float32)
    in_spatial = image.shape[1:]

    src_slices = []
    dst_slices = []
    for current, target in zip(in_spatial, target_shape):
        if current >= target:
            start = (current - target) // 2
            src_slices.append(slice(start, start + target))
            dst_slices.append(slice(0, target))
        else:
            start = (target - current) // 2
            src_slices.append(slice(0, current))
            dst_slices.append(slice(start, start + current))

    output[(slice(None), *dst_slices)] = image[(slice(None), *src_slices)]
    return output


def main() -> None:
    args = parse_args()
    patch_size = tuple(args.patch_size)
    case_path = list_case_paths(args.data_dir)[0]
    data = np.load(case_path)
    image = data["image"].astype(np.float32)
    image = center_crop_or_pad(image, patch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=args.feature_size,
        spatial_dims=3,
        use_v2=not args.disable_v2,
    ).to(device)
    model.eval()

    batch = torch.from_numpy(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch)

    print(f"case={Path(case_path).stem}")
    print(f"device={device}")
    print(f"input_shape={tuple(batch.shape)}")
    print(f"output_shape={tuple(logits.shape)}")
    print(f"output_dtype={logits.dtype}")
    print("smoke test passed")


if __name__ == "__main__":
    main()

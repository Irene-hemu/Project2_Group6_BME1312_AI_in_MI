from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from task3_swinunetr.dataset import (
    REGION_NAMES,
    BraTS3DPatchDataset,
    BraTS3DVolumeDataset,
    list_case_paths,
    split_case_paths,
)

try:
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceCELoss
    from monai.metrics import HausdorffDistanceMetric
    from monai.networks.nets import SwinUNETR
except ImportError as exc:
    raise SystemExit(
        "MONAI is not installed. Install it with `pip install monai` in a compatible environment first."
    ) from exc


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MONAI SwinUNETR on BraTS 3D NPZ volumes.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file. CLI flags override config values.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="outputs_task1/processed_3d",
        help="Directory containing Task 1 3D .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="task3_swinunetr/runs/swinunetr_v2_baseline",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-size", type=int, default=48)
    parser.add_argument("--samples-per-case", type=int, default=4)
    parser.add_argument("--foreground-prob", type=float, default=0.8)
    parser.add_argument("--sw-batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--infer-overlap", type=float, default=0.5)
    parser.add_argument("--disable-v2", action="store_true", help="Use original SwinUNETR instead of V2 blocks.")
    parser.add_argument("--disable-amp", action="store_true", help="Turn off mixed precision training.")
    parser.add_argument("--skip-hd95", action="store_true", help="Skip HD95 computation during validation.")
    parser.add_argument("--use-checkpoint", action="store_true", help="Enable gradient checkpointing in SwinUNETR.")
    parser.add_argument("--train-augment", action="store_true", help="Enable patch-level data augmentation.")
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--intensity-scale", type=float, default=0.1)
    parser.add_argument("--intensity-shift", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-train-cases", type=int, default=None)
    parser.add_argument("--max-val-cases", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true", help="Disable cuDNN benchmark for reproducibility.")
    return parser


def parse_args() -> argparse.Namespace:
    parser = create_parser()
    known_args, _ = parser.parse_known_args()
    if known_args.config:
        config_path = Path(known_args.config)
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        parser.set_defaults(**config_data)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_patch_size_is_valid(patch_size: tuple[int, int, int]) -> None:
    for dim in patch_size:
        if dim % 32 != 0:
            raise ValueError(
                f"Patch size {patch_size} is invalid for SwinUNETR. Each dimension should be divisible by 32."
            )


def build_model(feature_size: int, use_v2: bool, use_checkpoint: bool) -> SwinUNETR:
    return SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        spatial_dims=3,
        use_v2=use_v2,
    )


def dice_per_region(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred = (torch.sigmoid(logits) >= threshold).float()
    reduce_dims = tuple(range(2, pred.ndim))
    intersection = (pred * target).sum(dim=reduce_dims)
    denominator = pred.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    return (2.0 * intersection + 1e-5) / (denominator + 1e-5)


def mean_without_nan(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def train_one_epoch(
    model: SwinUNETR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DiceCELoss,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    return running_loss / max(1, len(loader))


@torch.no_grad()
def validate(
    model: SwinUNETR,
    loader: DataLoader,
    device: torch.device,
    patch_size: tuple[int, int, int],
    infer_overlap: float,
    sw_batch_size: int,
    compute_hd95: bool,
) -> dict[str, float | list[float]]:
    model.eval()
    dice_by_region: list[list[float]] = [[] for _ in REGION_NAMES]
    hd95_by_region: list[list[float]] = [[] for _ in REGION_NAMES]
    hd95_metric = None

    if compute_hd95:
        hd95_metric = HausdorffDistanceMetric(
            include_background=True,
            percentile=95,
            reduction="mean_batch",
        )

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        logits = sliding_window_inference(
            inputs=images,
            roi_size=patch_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=infer_overlap,
        )
        dice_values = dice_per_region(logits, targets).mean(dim=0).detach().cpu().tolist()

        for idx, value in enumerate(dice_values):
            dice_by_region[idx].append(float(value))

        if hd95_metric is not None:
            preds = (torch.sigmoid(logits) >= 0.5).float()
            hd95_metric(y_pred=preds, y=targets)
            hd95_values = hd95_metric.aggregate().detach().cpu().tolist()
            hd95_metric.reset()
            for idx, value in enumerate(hd95_values):
                hd95_by_region[idx].append(float(value))

    dice_mean = [mean_without_nan(values) for values in dice_by_region]
    result: dict[str, float | list[float]] = {
        "dice_regions": dice_mean,
        "dice_mean": mean_without_nan(dice_mean),
    }

    if compute_hd95:
        hd95_mean = [mean_without_nan(values) for values in hd95_by_region]
        result["hd95_regions"] = hd95_mean
        result["hd95_mean"] = mean_without_nan(hd95_mean)

    return result


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def limit_case_paths(case_paths: list[Path], max_cases: int | None) -> list[Path]:
    if max_cases is None:
        return case_paths
    if max_cases <= 0:
        raise ValueError(f"max_cases must be positive when provided, got {max_cases}")
    return case_paths[:max_cases]


def main() -> None:
    args = parse_args()
    patch_size = tuple(args.patch_size)
    ensure_patch_size_is_valid(patch_size)
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not args.deterministic

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_paths = list_case_paths(args.data_dir)
    train_paths, val_paths = split_case_paths(case_paths, args.val_ratio, args.seed)
    train_paths = limit_case_paths(train_paths, args.max_train_cases)
    val_paths = limit_case_paths(val_paths, args.max_val_cases)

    train_dataset = BraTS3DPatchDataset(
        case_paths=train_paths,
        patch_size=patch_size,
        samples_per_case=args.samples_per_case,
        foreground_prob=args.foreground_prob,
        augment=args.train_augment,
        flip_prob=args.flip_prob,
        intensity_scale=args.intensity_scale,
        intensity_shift=args.intensity_shift,
        noise_std=args.noise_std,
    )
    val_dataset = BraTS3DVolumeDataset(case_paths=val_paths)

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = args.persistent_workers
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(1, args.num_workers)),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        feature_size=args.feature_size,
        use_v2=not args.disable_v2,
        use_checkpoint=args.use_checkpoint,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, lambda_dice=0.7, lambda_ce=0.3)
    amp_enabled = (not args.disable_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    config = vars(args).copy()
    config["patch_size"] = list(patch_size)
    config["train_cases"] = len(train_paths)
    config["val_cases"] = len(val_paths)
    save_json(output_dir / "config.json", config)
    save_json(
        output_dir / "splits.json",
        {
            "train_cases": [path.stem for path in train_paths],
            "val_cases": [path.stem for path in val_paths],
        },
    )

    history: list[dict[str, float | int | list[float]]] = []
    best_dice = -1.0

    print(f"Device: {device}")
    print(f"Train cases: {len(train_paths)} | Val cases: {len(val_paths)}")
    print(f"Using SwinUNETR V2 blocks: {not args.disable_v2}")
    print(f"Patch size: {patch_size} | Feature size: {args.feature_size} | AMP: {amp_enabled}")
    print(f"Gradient checkpointing: {args.use_checkpoint} | Train augment: {args.train_augment}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, amp_enabled, scaler)
        metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            patch_size=patch_size,
            infer_overlap=args.infer_overlap,
            sw_batch_size=args.sw_batch_size,
            compute_hd95=not args.skip_hd95,
        )

        epoch_result: dict[str, float | int | list[float]] = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }
        history.append(epoch_result)
        save_json(output_dir / "history.json", {"epochs": history})

        dice_regions = epoch_result["dice_regions"]
        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"dice_mean={epoch_result['dice_mean']:.4f} | "
            f"dice_regions={dice_regions}"
        )

        if float(epoch_result["dice_mean"]) > best_dice:
            best_dice = float(epoch_result["dice_mean"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": epoch_result,
                    "config": config,
                },
                output_dir / "best_model.pt",
            )

    print(f"Best validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()

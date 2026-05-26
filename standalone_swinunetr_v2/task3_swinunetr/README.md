# Task 3 SwinUNETR Starter

This folder gives you a practical Task 3 baseline around MONAI's `SwinUNETR` without copying the full MONAI repository into your coursework project.

## Why this approach

- Your project already contains 3D BraTS volumes under `outputs_task1/processed_3d`.
- `SwinUNETR` is already implemented and maintained in MONAI.
- The V2-style variant is available through `use_v2=True`, so you can align your baseline with the newer SwinV2-flavored design while keeping the rest of the code small.

## What is included

- `dataset.py`: loads your Task 1 `.npz` volumes, samples foreground-biased 3D patches, and applies lightweight patch augmentation.
- `train.py`: trains MONAI `SwinUNETR` on `WT/TC/ET` region targets and validates with sliding-window inference.
- `configs/local_debug_v2.json`: tiny local config for script validation.
- `configs/cluster_swinunetr_v2_single_gpu.json`: single-GPU cluster config aimed at a high-memory node.

## Recommended environment

Your working `swin` environment has already been verified with:

- `torch==2.8.0+cu128`
- `monai==1.5.2`
- CUDA available locally

MONAI is much smaller to install as a package than to vendor from source. The training code here is written against the installed `MONAI 1.5.2` API.

Suggested safe environment:

- Python `3.10` or `3.11`
- PyTorch version officially supported by the MONAI release you install
- `monai`, `numpy`, `scipy`, `tqdm`, `einops`

## Install

```bash
pip install monai numpy scipy tqdm einops
```

If that does not work in your current Anaconda base env, create a fresh env and install there instead.

## Run

From the project root:

```powershell
python -m task3_swinunetr.train `
  --config task3_swinunetr/configs/local_debug_v2.json
```

Single-GPU cluster run:

```powershell
python -m task3_swinunetr.train `
  --config task3_swinunetr/configs/cluster_swinunetr_v2_single_gpu.json
```

## Important choices

- Use `processed_3d`, not `processed_2d`, because SwinUNETR is most meaningful as a 3D Task 3 model.
- Keep `patch-size` divisible by `32`. Good starting values are `96 96 96` for local debugging and `128 128 128` for a larger GPU.
- Leave V2 enabled by default. Add `--disable-v2` only if you want to compare against the original SwinUNETR blocks.
- The cluster config is intentionally more aggressive: `feature_size=48`, `batch_size=2`, `patch_size=128^3`, patch augmentation on, and AMP on.
- Validation splits are saved to `splits.json` inside the run directory for reproducibility.
- `skip_hd95=true` is recommended during long training runs; you can turn it back on later for final evaluation.

## Good report story for Task 3

You can now structure Task 3 as:

1. `2D U-Net` baseline from Task 2
2. `3D SwinUNETR v2` advanced baseline
3. Dice and HD95 comparison on `WT/TC/ET`
4. Worst-case analysis on the 5 hardest validation cases

That is a much cleaner story than trying to reimplement the whole MONAI codebase from scratch.

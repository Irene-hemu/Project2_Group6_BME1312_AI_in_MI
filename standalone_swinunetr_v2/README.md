# Standalone SwinUNETR V2 Package

This directory is a clean export of the Task 3 `SwinUNETR v2` training package.

## Layout

- `task3_swinunetr/`: training code, smoke test, configs, and dependency list

## Expected project-relative data path

The provided configs assume your BraTS Task 1 outputs live at:

```text
outputs_task1/processed_3d
```

If you place this directory inside another repository, keep that relative path or update the JSON config files.

## Quick start

PowerShell local debug:

```powershell
python -m task3_swinunetr.train `
  --config task3_swinunetr/configs/local_debug_v2.json
```

PowerShell smoke test:

```powershell
python -m task3_swinunetr.smoke_test
```

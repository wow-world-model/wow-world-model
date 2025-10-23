# IDM_release

A concise implementation of image-to-action models using DINO and Co-Tracker features. The project provides model definitions, dataset utilities (with optional precomputation), training and simple inference scripts to predict 7-DOF robot joint actions from consecutive image frames (optionally with a text instruction).

## Repository structure
- `models/`
  - `idm_models.py` — model definitions (DinoIDM, Dino3DFlowIDM, ResNetIDM)
- `datasets/`
  - `dataset.py` — datasets: folder-based, HDF5-based, and DINO+Flow precomputed feature dataset
- `train.py` — training entry (uses RLBenchFLowFeatureDataset)
- `infer.py` — simplified inference script; accepts h5/hdf5 files and outputs JSON results
- `checkpoints/`, `logs/` — created at runtime

## Key features
- Uses timm DINO ViT encoder for image features
- Optional Co-Tracker flow features (Dino3DFlowIDM)
- Precompute and cache DINO + Co-Tracker features to speed up training
- Handles DataParallel checkpoints (removes `module.` prefix automatically)

## Requirements
- Python >= 3.8
- torch, torchvision
- timm
- h5py
- pillow
- tqdm
- tensorboard
- numpy
- Internet access for torch.hub to download `facebookresearch/co-tracker` (if used)
- (Optional) Video-Depth-Anything if dataset utilities rely on it
- (Optional) If you want to perform robotic arm segmentation, please refer to the environment setup of SAM2.

Install example:
```bash
python -m pip install torch torchvision timm h5py pillow tqdm tensorboard numpy
# co-tracker will be fetched via torch.hub; ensure network access
```

## Data formats
Supported datasets must match the code expectations:

1. HDF5 / .hdf5 files
   - `text`: instruction (bytes or str)
   - `action`: array with shape (T, 7)
   - `observations/image`: array with shape (T, H, W, 3) (commonly 224x224)

2. Folder-based episodes (`RLBenchDataset`)
   - Each episode directory contains `images/*.png`, `actions.npy`, `text.txt`
   - Root contains `tasks.json` mapping text -> id

Note: `RLBenchFLowFeatureDataset` may precompute DINO and Co-Tracker flow features and save `features.pth`. This step can be slow and requires significant memory and disk space.

## Usage

Training example:
```bash
python train.py \
  --data_path /path/to/dataset_root \
  --save_path ./checkpoints \
  --batch_size 16 \
  --total_steps 20000 \
  --output_dim 7 \
  --model dinov2_flow
```
- Supports resuming with `--resume_from`.

Inference example:
```bash
python infer.py \
  --h5_files /path/to/file1.hdf5 /path/to/file2.hdf5 \
  --model_path ./checkpoints/checkpoint_1000.pth \
  --output_path ./simple_inference_results.json \
  --device cuda \
  --max_actions 200 \
  --model dinov2_flow \
  --output_dim 7
```
- `infer.py` loads checkpoints (removes DataParallel `module.` prefix if present) and uses any saved `text_map` to map instructions to IDs.
- Output: JSON list of records containing file, frame index, and predicted joint values.

Sanitized command examples (replace placeholders with actual paths):

Replay / inference (example):
```bash
python /path/to/replay.py \
  --h5_files /path/to/hdf5/example_file.hdf5 \
  --model_path /path/to/checkpoints/checkpoint_latest.pth \
  --output_path /path/to/output/replay_results.json \
  --model dinov2_flow \
  --output_dim 7
```

Long-run training example:
```bash
python /path/to/train_flow_by.py \
  --data_path /path/to/dataset/keyframe_h5 \
  --save_path /path/to/save/checkpoints/dinov2-flow-experiment \
  --model dinov2_flow \
  --total_steps 1000000 \
  --batch_size 64 \
  --gradient_accumulation 5 \
  --save_every 1000 \
  --lr 2e-4
```

## Models
- DinoIDM — ViT (DINOv2) encoder, two-frame input, with optional text embedding.
- Dino3DFlowIDM — DINO image features + Co-Tracker flow trajectories (tracks × 2 → flow feature), has `infer_forward` for runtime flow computation and `forward` for training with precomputed features.
- ResNetIDM — ResNet-based alternative.

## Notes and troubleshooting
- Use `--device cpu` if CUDA is unavailable. The scripts try to detect and fallback when `cuda` is requested but not available.
- Co-Tracker is downloaded via torch.hub on first use; this requires network access.
- Feature precomputation (`features.pth`) can be large; ensure sufficient disk and memory.
- `datasets/dataset.py` contains a hard-coded import path for Video-Depth-Anything; adjust if needed.

## License
This repository is licensed under the Apache License, Version 2.0.

See: https://www.apache.org/licenses/LICENSE-2.0

SPDX-License-Identifier: Apache-2.0

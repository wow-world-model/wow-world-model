# 🎥 WAN Video Generation - Simplified DiffSynth-Studio

This guide provides a step-by-step walkthrough for setting up the environment and running inference using our **simplified DiffSynth-Studio** for WAN video generation.

> **📌 Note:** This is a **simplified version** of DiffSynth-Studio containing **only WAN video generation code**. All other models (Stable Diffusion, Flux, CogVideo, HunyuanVideo, etc.) have been removed to create a minimal, focused codebase.

---

## ✨ What's Included

This simplified version includes only:
- **WAN video models** (DiT, VAE, text encoder, image encoder, motion controller, VACE)
- **WAN pipeline** for video generation
- **Flow matching scheduler**
- **Distributed training support** (USP - Unified Sequence Parallel)
- **VRAM management** utilities
- **Essential utilities** (attention, tiler, LoRA support)

---

## 🚀 Quick Start

Get up and running in 3 steps:

```bash
# 1. Install dependencies
cd dit_models/DiffSynth-Studio
pip install -e .
cd ../..

# 2. Download checkpoints
huggingface-cli download WoW-world-model/WoW-1-Wan-14B-600k \
  --local-dir dit_models/checkpoints/WoW-1-Wan-14B-600k

# 3. Run the demo (optimized for H800 80GB GPU)
python demo/wan_infer_demo.py \
  --checkpoint_folder dit_models/checkpoints/WoW-1-Wan-14B-600k \
  --persistent_param_gb 70
```

Then open your browser to `http://localhost:7860` and start generating videos!

> 💡 **Note:** Default settings are optimized for H800/H100 (80GB VRAM). See [Performance Optimization](#-performance-optimization) for other GPU configurations.

---

## 📦 Environment Setup

### 1. Repository Structure
The DiffSynth-Studio code is already included in this repository at:
```
dit_models/wan-simple  (Simplified WAN-only version)
```

### 2. Create Conda Environment
```bash
conda create -n wowwan python=3.10
conda activate wowwan
```

### 3. Install Dependencies
Navigate to the wan-simple directory and install:
```bash
cd dit_models/wan-simple
pip install -e .
```

> ℹ️ **Reference:** For more details about WAN video models, see the [original DiffSynth-Studio WAN documentation](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/README.md)

---

## 📁 Model Preparation

### Required Checkpoint Folder Structure

The checkpoint folder must contain the following base model files:

```bash
dit_models/checkpoints/WoW-1-Wan-14B-600k/
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  # CLIP image encoder
├── models_t5_umt5-xxl-enc-bf16.pth                          # T5 text encoder
├── Wan2.1_VAE.pth                                           # VAE encoder/decoder
├── diffusion_pytorch_model-00001-of-00007.safetensors      # Base DiT model (part 1)
├── diffusion_pytorch_model-00002-of-00007.safetensors      # Base DiT model (part 2)
├── diffusion_pytorch_model-00003-of-00007.safetensors      # Base DiT model (part 3)
├── diffusion_pytorch_model-00004-of-00007.safetensors      # Base DiT model (part 4)
├── diffusion_pytorch_model-00005-of-00007.safetensors      # Base DiT model (part 5)
├── diffusion_pytorch_model-00006-of-00007.safetensors      # Base DiT model (part 6)
├── diffusion_pytorch_model-00007-of-00007.safetensors      # Base DiT model (part 7)
└── WoW_video_dit.pt                                        # Custom DiT checkpoint (optional)
```

### Download Model Checkpoints

**Option 1: Download complete checkpoint folder from Hugging Face:**

```bash
# Install Hugging Face CLI if not already installed
pip install huggingface-hub

# Download the complete checkpoint folder
huggingface-cli download WoW-world-model/WoW-1-Wan-14B-600k \
  --local-dir dit_models/checkpoints/WoW-1-Wan-14B-600k
```

**Option 2: Manual download:**

1. Visit [WoW-world-model/WoW-1-Wan-14B-600k on Hugging Face](https://huggingface.co/WoW-world-model/WoW-1-Wan-14B-600k)
2. Download all required files listed above
3. Place them in `dit_models/checkpoints/WoW-1-Wan-14B-600k/`

### Custom Checkpoint (Optional)

If you have a custom-trained DiT checkpoint:
1. Place your checkpoint file in the checkpoint folder
2. Use the `--custom_checkpoint` argument to specify its filename
3. Example: `--custom_checkpoint my_custom_model.pt`

> 📚 **More Checkpoints:** Find additional checkpoints at [WoW-world-model on Hugging Face](https://huggingface.co/WoW-world-model). We regularly update with new checkpoints.

---

## ▶️ How to Run Inference

Run the inference demo with a fancy Gradio web interface from the repository root:

```bash
python demo/wan_infer_demo.py \
  --checkpoint_folder dit_models/checkpoints/WoW-1-Wan-14B-600k \
  --custom_checkpoint WoW_video_dit.pt \
  --gpu 0 \
  --port 7860
```

### Command Line Arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint_folder` | str | `dit_models/checkpoints/WoW-1-Wan-14B-600k` | Path to folder containing base model files (CLIP, T5, VAE, DiT) |
| `--custom_checkpoint` | str | `WoW_video_dit.pt` | Filename of custom DiT checkpoint (optional) |
| `--gpu` | int | `0` | GPU device ID to use |
| `--port` | int | `7860` | Port for the Gradio web interface |
| `--persistent_param_gb` | int | `70` | GB of model params to keep in GPU (70GB for H800, adjust for your GPU) |
| `--enable_vram_management` | flag | `True` | Enable VRAM management (recommended for memory optimization) |
| `--no_vram_management` | flag | `False` | Disable VRAM management (use full GPU memory) |
| `--share` | flag | `False` | Create a public Gradio share link |

### Example Commands:

**Basic usage with default settings:**
```bash
python demo/wan_infer_demo.py --checkpoint_folder dit_models/checkpoints/WoW-1-Wan-14B-600k
```

**Use a different custom checkpoint:**
```bash
python demo/wan_infer_demo.py \
  --checkpoint_folder dit_models/checkpoints/WoW-1-Wan-14B-600k \
  --custom_checkpoint my_custom_model.pt \
  --gpu 0
```

**Without custom checkpoint (use base DiT model only):**
```bash
python demo/wan_infer_demo.py \
  --checkpoint_folder dit_models/checkpoints/WoW-1-Wan-14B-600k \
  --custom_checkpoint "" \
  --gpu 0
```

**Create a public share link:**
```bash
python demo/wan_infer_demo.py \
  --checkpoint_folder dit_models/checkpoints/WoW-1-Wan-14B-600k \
  --share
```

**Optimize for H800 GPU (80GB VRAM):**
```bash
python demo/wan_infer_demo.py \
  --checkpoint_folder dit_models/checkpoints/WoW-1-Wan-14B-600k \
  --persistent_param_gb 70 \
  --gpu 0
```

**Optimize for A100 GPU (40GB VRAM):**
```bash
python demo/wan_infer_demo.py \
  --checkpoint_folder dit_models/checkpoints/WoW-1-Wan-14B-600k \
  --persistent_param_gb 35 \
  --gpu 0
```

**Disable VRAM management (maximum speed, requires large GPU):**
```bash
python demo/wan_infer_demo.py \
  --checkpoint_folder dit_models/checkpoints/WoW-1-Wan-14B-600k \
  --no_vram_management
```

---

## ⚡ Performance Optimization

### GPU-Specific Settings

The `--persistent_param_gb` parameter controls how much model data stays in GPU memory. Higher values = faster generation but more VRAM usage.

**Recommended settings by GPU:**

| GPU Model | VRAM | Recommended `--persistent_param_gb` | Notes |
|-----------|------|-------------------------------------|-------|
| H800 | 80GB | `70` | Optimal for maximum speed |
| H100 | 80GB | `70` | Optimal for maximum speed |
| A100 (80GB) | 80GB | `70` | Optimal for maximum speed |
| A100 (40GB) | 40GB | `35` | Balance speed and stability |
| A6000 | 48GB | `42` | Good balance |
| RTX 4090 | 24GB | `20` | May need tiled mode for long videos |
| RTX 3090 | 24GB | `20` | May need tiled mode for long videos |

### Speed Optimization Tips

For **maximum speed** on H800/H100:
1. Use `--persistent_param_gb 70` (default)
2. Keep VRAM management enabled (default)
3. Use tiled mode for videos > 41 frames
4. Ensure no other processes are using the GPU

For **maximum quality**:
1. Increase inference steps to 80-100
2. Use higher frame counts (60-100)
3. Use tiled mode for better consistency

For **memory-constrained GPUs**:
1. Lower `--persistent_param_gb` value
2. Enable tiled mode in the web interface
3. Reduce number of frames (25-30)
4. Reduce inference steps (30-40)

### Monitoring GPU Usage

To verify your H800 is fully utilized, monitor GPU usage during generation:

```bash
# In a separate terminal, run:
watch -n 1 nvidia-smi

# Or for more detailed info:
nvitop
```

**What to look for:**
- **GPU Memory Usage**: Should be close to your `persistent_param_gb` value (e.g., ~70GB on H800)
- **GPU Utilization**: Should be 95-100% during generation
- **Power Usage**: Should be near max TDP (700W for H800)

If GPU utilization is low:
- Check if other processes are using the GPU
- Verify VRAM management is enabled
- Ensure `persistent_param_gb` is set appropriately
- Try disabling VRAM management with `--no_vram_management` for maximum speed (if you have enough memory)

---

## 🎨 Demo Features

The Gradio web interface includes:

- **Modern UI Theme**: Beautiful gradient design with blue/purple colors
- **Image Preview**: Upload images or videos and see previews instantly
- **Video Support**: Automatically extracts first frame from uploaded videos
- **Advanced Settings**: Control inference steps, number of frames, random seed, and more
- **Tiled Mode**: Enable for better memory efficiency with long videos
- **Real-time Status**: See generation progress and status messages
- **Responsive Layout**: Clean, two-column layout with collapsible settings

### Interface Controls:

1. **Upload Input**: Upload an image (.jpg, .png) or video (.mp4, .avi, .mov)
2. **Text Prompt**: Describe the action or scene you want to generate
3. **Advanced Settings** (expandable accordion):
   - **Inference Steps**: 1-100 (higher = better quality, slower)
   - **Number of Frames**: 1-100 (controls video length)
   - **Random Seed**: For reproducible results
   - **GPU ID**: Select which GPU to use
   - **Tiled Mode**: Reduce memory usage for longer videos
4. **Generate Button**: Click to start video generation
5. **Output Panel**: View generation status and play the generated video

### Example Prompts:

- "A person picking up a cup from the table"
- "A car driving down a city street at sunset"
- "A cat jumping onto a sofa"
- "A robot arm assembling a mechanical part"
- "Water pouring into a glass"

---

## 🗂️ Simplified Codebase Details

### What Was Removed:
- ❌ All Stable Diffusion models (SD, SDXL, SD3)
- ❌ Flux, CogVideo, HunyuanDiT, HunyuanVideo models
- ❌ StepVideo, SVD, OmniGen models
- ❌ ControlNet and IPAdapter implementations
- ❌ Image post-processing extensions (RIFE, ESRGAN)
- ❌ Training datasets and trainers
- ❌ All non-WAN pipelines, prompters, and schedulers
- ❌ Unused tokenizer configurations

### What Remains (Core WAN Components):

#### Models (`diffsynth/models/`):
- `wan_video_dit.py` - WAN DiT transformer model
- `wan_video_vae.py` - WAN VAE encoder/decoder
- `wan_video_text_encoder.py` - Text encoding
- `wan_video_image_encoder.py` - Image encoding
- `wan_video_motion_controller.py` - Motion control
- `wan_video_vace.py` - VACE model
- Supporting files: `attention.py`, `utils.py`, `tiler.py`, `lora.py`, `model_manager.py`, `downloader.py`

#### Pipelines (`diffsynth/pipelines/`):
- `wan_video.py` - Main WAN video generation pipeline
- `base.py` - Base pipeline class

#### Prompters (`diffsynth/prompters/`):
- `wan_prompter.py` - WAN-specific text prompter
- `base_prompter.py` - Base prompter class

#### Schedulers (`diffsynth/schedulers/`):
- `flow_match.py` - Flow matching scheduler for WAN

#### Support Modules:
- `distributed/` - Multi-GPU distributed training support
- `vram_management/` - Memory optimization utilities
- `configs/` - Simplified model configurations

---

## ⚠️ Troubleshooting

### GPU Memory Issues
WAN video generation requires significant GPU memory. If you encounter OOM (Out of Memory) errors:

1. **Adjust Persistent Params**: Lower the `--persistent_param_gb` value based on your GPU (see [Performance Optimization](#-performance-optimization))
   - Example: `--persistent_param_gb 35` for 40GB GPUs
   - Example: `--persistent_param_gb 20` for 24GB GPUs
2. **Enable Tiled Mode**: Check the "Use Tiled Mode" checkbox in the web interface
3. **Reduce Number of Frames**: Generate shorter videos (e.g., 25-30 frames instead of 41)
4. **Lower Inference Steps**: Use fewer steps (e.g., 30-40 instead of 50)
5. **Close Other Applications**: Free up GPU memory by closing other GPU-intensive programs


## 📝 License

This simplified version maintains the same license as the original DiffSynth-Studio repository.

## 🙏 Acknowledgments

- **DiffSynth-Studio**: Original codebase by ModelScope
- **WAN Video Models**: For state-of-the-art video generation capabilities

For issues or questions, please refer to the original [DiffSynth-Studio repository](https://github.com/modelscope/DiffSynth-Studio).


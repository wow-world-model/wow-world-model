#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WoW World Model - 1.3B I2V Inference Demo
"""

import os
import tempfile
import torch
from PIL import Image
import gradio as gr
from pathlib import Path
import cv2
import argparse
from diffsynth import ModelManager, WanVideoPipeline, save_video
import inspect

torch.serialization.add_safe_globals(['set', 'OrderedDict', 'builtins.set'])


def extract_first_frame(video_path):
    """Extract the first frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read video: {video_path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def preview_uploaded_file(file_path):
    """
    Process uploaded file for preview.
    Returns PIL Image for both image and video files.
    """
    if file_path is None:
        return None

    file_path = str(file_path)

    # Check if it's a video file
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        try:
            # Extract first frame from video
            return extract_first_frame(file_path)
        except Exception as e:
            print(f"Error extracting frame from video: {e}")
            return None
    else:
        # It's an image file, load and return it
        try:
            return Image.open(file_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            return None


def ensure_in_channels_32_before_lora(model=None):
    """Ensure model supports 32 input channels (x:16 + y:32) before LoRA injection."""
    if hasattr(model, 'patch_embedding'):
        if hasattr(model.patch_embedding, 'module'):
            patch_emb = model.patch_embedding.module
        else:
            patch_emb = model.patch_embedding
        current_in_dim = patch_emb.weight.shape[1]
        if current_in_dim != 36:
            print(f"[INFO] Expanding patch_embedding input channels from {current_in_dim} to 36 before LoRA")
            original_weight = patch_emb.weight.data
            out_dims = list(original_weight.shape)
            out_channels = out_dims[0]
            in_channels = out_dims[1]
            remaining_shape = out_dims[2:]
            new_weight = torch.zeros((out_channels, 36, *remaining_shape), dtype=original_weight.dtype, device=original_weight.device)
            new_weight[:, :in_channels] = original_weight
            torch.nn.init.normal_(new_weight[:, in_channels:], mean=0.0, std=0.02)
            patch_emb.weight.data = new_weight
        else:
            print(f"[INFO] patch_embedding input channels already 36")
    else:
        print("[WARN] denoising model has no patch_embedding; please verify architecture")


def build_pipeline(gpu_id=0, base_model_folder=None, checkpoint_path=None, enable_vram_management=True, persistent_param_gb=60):
    """
    Build WAN 1.3B I2V pipeline from base models and optional checkpoint.

    Args:
        gpu_id: GPU device ID
        base_model_folder: Path to folder containing base 1.3B model files. Expected structure:
            - models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
            - models_t5_umt5-xxl-enc-bf16.pth
            - Wan2.1_VAE.pth
            - diffusion_pytorch_model.safetensors
        checkpoint_path: Path to custom checkpoint folder (e.g., "output/wan_1.3B_full_2000k_2/checkpoints/wan-epoch=93-train_loss=0.0491.ckpt")
        enable_vram_management: Enable VRAM management for memory optimization
        persistent_param_gb: Number of GB to keep persistent in GPU (default 60GB for 1.3B)
    """
    device = f"cuda:{gpu_id}"
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")  # Load on CPU first, then move to GPU
    
    base_model_folder = Path(base_model_folder)
    if not base_model_folder.exists():
        raise FileNotFoundError(f"Base model folder does not exist: {base_model_folder}")

    # Define model paths for 1.3B
    # Use I2V-14B CLIP (for better image understanding) and 1.3B DiT
    # CLIP model should come from I2V-14B model (not from 1.3B model)
    clip_model_path = base_model_folder / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    dit_model_path = base_model_folder / "diffusion_pytorch_model.safetensors"
    t5_model_path = base_model_folder / "models_t5_umt5-xxl-enc-bf16.pth"
    vae_model_path = base_model_folder / "Wan2.1_VAE.pth"

    # Load CLIP model
    print("📦 Loading CLIP image encoder...")
    mm.load_models([str(clip_model_path)], torch_dtype=torch.float32)

    # Load DiT, T5, and VAE models
    print("📦 Loading DiT, T5 text encoder and VAE...")
    mm.load_models(
        [str(dit_model_path), str(t5_model_path), str(vae_model_path)],
        torch_dtype=torch.bfloat16,
    )

    # Check for custom checkpoint
    if checkpoint_path:
        checkpoint_path_obj = Path(checkpoint_path)
        if checkpoint_path_obj.exists():
            print(f"🎯 Loading custom checkpoint: {checkpoint_path}")
            checkpoint_file = checkpoint_path_obj / "checkpoint" / "mp_rank_00_model_states.pt"
            
            if not checkpoint_file.exists():
                print(f"⚠️  Checkpoint file not found: {checkpoint_file}")
                print("   Continuing with base model...")
            else:
                try:
                    state_dict = torch.load(str(checkpoint_file), map_location="cpu")
                    dit_model = mm.fetch_model("wan_video_dit")
                    if dit_model is not None:
                        ensure_in_channels_32_before_lora(dit_model)
                        dit_model.load_state_dict(state_dict, strict=False)
                        dit_model.has_image_input = True
                        print("✅ Custom checkpoint loaded successfully")
                except Exception as e:
                    print(f"⚠️  Failed to load custom checkpoint: {e}")
                    print("   Continuing with base model...")
        else:
            print(f"⚠️  Checkpoint path does not exist: {checkpoint_path}")
            print("   Continuing with base model...")

    # Build pipeline
    pipe = WanVideoPipeline.from_model_manager(mm, torch_dtype=torch.bfloat16, device=device)

    # Configure VRAM management for optimal performance
    if enable_vram_management:
        num_persistent_params = int(persistent_param_gb * 10**9)
        pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_params)
        print(f"✅ VRAM management enabled: {persistent_param_gb}GB persistent params")
    else:
        print("⚠️  VRAM management disabled (may cause OOM on large models)")

    # Enable image input for I2V mode
    pipe.denoising_model().has_image_input = True
    
    print(f"✅ Pipeline built successfully on {device}")
    return pipe


# Global model variable
pipe = None


def generate_video(prompt, input_file, gpu_id, steps=50, seed=42, tiled=True, num_frames=81):
    """Generate video from input image and text prompt."""
    global pipe

    if not prompt or input_file is None:
        return "❌ Error: Prompt and input image are required", None

    if pipe is None:
        return "❌ Error: Model not loaded, please check the startup logic", None

    # Extract image from input file
    # input_file is a filepath string from gr.File
    if isinstance(input_file, str) and input_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        input_image = extract_first_frame(input_file)
    else:
        # For image files, load directly
        input_image = Image.open(input_file).convert("RGB")

    # Generate video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        output_path = tmpfile.name

    video = pipe(
        prompt=prompt,
        negative_prompt="low quality, distorted, ugly, bad anatomy",
        input_image=input_image,
        num_inference_steps=steps,
        seed=seed,
        tiled=tiled,
        num_frames=num_frames,
    )

    save_video(video, output_path, fps=15, quality=5)
    return "✅ Generation successful!", output_path


def build_interface():
    """Build Gradio interface with custom theme."""

    # Create custom theme with cool, modern design
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        body_background_fill_dark="linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
        button_primary_background_fill="linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
        button_primary_background_fill_hover="linear-gradient(90deg, #764ba2 0%, #667eea 100%)",
        button_primary_text_color="white",
        block_title_text_weight="600",
        block_label_text_weight="600",
        input_background_fill="#ffffff",
        input_background_fill_dark="#2d3748",
    )

    with gr.Blocks(
        title="WoW Video Generation Demo (1.3B I2V)",
        theme=theme,
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .main-header {
            text-align: center;
            color: #ffffff !important;
            font-size: 2.8em !important;
            font-weight: 800 !important;
            margin-bottom: 0.3em;
            text-shadow: 0 4px 12px rgba(0, 0, 0, 0.3), 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .subtitle {
            text-align: center;
            color: #e2e8f0;
            font-size: 1.15em;
            margin-bottom: 2em;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        """
    ) as demo:
        gr.Markdown(
            """
            <h1 class="main-header">🎬 WoW World Generation Studio (1.3B I2V)</h1>
            <p class="subtitle">Transform images into robot manipulation videos with AI-powered world models</p>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # File upload component
                input_file = gr.File(
                    label="📁 Upload Input Image or Video",
                    file_types=["image", "video"],
                    type="filepath",
                )
                # Preview component
                input_preview = gr.Image(
                    label="📸 Preview (First Frame)",
                    interactive=False,
                    height=300,
                )
                prompt = gr.Textbox(
                    label="✨ Text Prompt",
                    placeholder="Describe the action or scene you want to generate, e.g., 'A Franka robot put the screw driver into the drawer'",
                    lines=3,
                )

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        steps = gr.Slider(
                            1, 100,
                            value=50,
                            step=1,
                            label="Inference Steps",
                        )
                        num_frames = gr.Slider(
                            1, 100,
                            value=81,
                            step=1,
                            label="Number of Frames",
                        )

                    with gr.Row():
                        seed = gr.Number(
                            label="Random Seed",
                            value=42,
                            precision=0,
                        )
                        gpu_id = gr.Number(
                            label="GPU ID",
                            value=0,
                            precision=0,
                        )

                    tiled = gr.Checkbox(
                        label="Use Tiled Mode (better memory efficiency)",
                        value=True,
                    )

                generate_btn = gr.Button(
                    "🚀 Generate Video",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                status = gr.Textbox(
                    label="📊 Status",
                    interactive=False,
                )
                output_video = gr.Video(
                    label="🎥 Generated Video",
                    format="mp4",
                )

        gr.Markdown(
            """
            ---
            ### 💡 Tips & Guidelines
            - **Input**: Upload an image (.jpg, .png) or video (.mp4, .avi, .mov) - preview shows first frame
            - **Prompt**: Write a detailed description of the action or scene you want to generate
            - **Steps**: Higher values (50-100) = better quality but slower generation
            - **Frames**: Control video length (more frames = longer video)
            - **Tiled Mode**: Enable to reduce memory usage for longer/higher resolution videos
            - **Seed**: Use same seed for reproducible results
            - **Model**: Using 1.3B I2V model for efficient image-to-video generation
            """
        )

        # Update preview when file is uploaded
        input_file.change(
            fn=preview_uploaded_file,
            inputs=[input_file],
            outputs=[input_preview],
        )

        # Generate video button
        generate_btn.click(
            fn=generate_video,
            inputs=[prompt, input_file, gpu_id, steps, seed, tiled, num_frames],
            outputs=[status, output_video],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WoW Video Generation Demo (1.3B I2V)")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--base_model_folder",
        type=str,
        default="dit_models/checkpoints/Wan2.1-T2V-1.3B",
        help="Path to folder containing 1.3B base model files"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to custom checkpoint folder (e.g., 'output/wan_1.3B_full_2000k_2/checkpoints/wan-epoch=93-train_loss=0.0491.ckpt')"
    )
    parser.add_argument(
        "--enable_vram_management",
        action="store_true",
        default=True,
        help="Enable VRAM management for memory optimization (recommended)"
    )
    parser.add_argument(
        "--no_vram_management",
        action="store_true",
        help="Disable VRAM management (use full GPU memory, may cause OOM)"
    )
    parser.add_argument(
        "--persistent_param_gb",
        type=int,
        default=60,
        help="GB of model parameters to keep persistent in GPU memory (default: 60GB for 1.3B model)"
    )
    parser.add_argument("--share", action="store_true", help="Create a public link")
    args = parser.parse_args()

    # Handle VRAM management flag logic
    enable_vram = args.enable_vram_management and not args.no_vram_management

    print("=" * 60)
    print(f"🎬 WoW Video Generation Demo (1.3B I2V)")
    print("=" * 60)
    print(f"📍 Base model folder: {args.base_model_folder}")
    if args.checkpoint:
        print(f"📦 Custom checkpoint: {args.checkpoint}")
    else:
        print(f"📦 Custom checkpoint: None (using base model)")
    print(f"🎮 GPU ID: {args.gpu}")
    print(f"🌐 Port: {args.port}")
    print(f"💾 VRAM Management: {'Enabled' if enable_vram else 'Disabled'}")
    if enable_vram:
        print(f"   Persistent Params: {args.persistent_param_gb}GB")
    print("=" * 60)
    print("⏳ Loading model, please wait...")

    pipe = build_pipeline(
        gpu_id=args.gpu,
        base_model_folder=args.base_model_folder,
        checkpoint_path=args.checkpoint,
        enable_vram_management=enable_vram,
        persistent_param_gb=args.persistent_param_gb
    )

    print("=" * 60)
    print("✅ Model loaded successfully!")
    print("🚀 Launching Gradio interface...")
    print("=" * 60)

    demo = build_interface()
    demo.launch(server_port=args.port, share=args.share)

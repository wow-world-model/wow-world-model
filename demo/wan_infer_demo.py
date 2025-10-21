#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tempfile
import torch
from PIL import Image
import gradio as gr
from pathlib import Path
import cv2
import argparse
from diffsynth import ModelManager, WanVideoPipeline, save_video

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


def build_pipeline(gpu_id=0, checkpoint_folder=None, custom_checkpoint_name=None,
                   enable_vram_management=True, persistent_param_gb=70):
    """
    Build WAN video pipeline from a checkpoint folder.

    Args:
        gpu_id: GPU device ID
        checkpoint_folder: Path to folder containing model files. Expected structure:
            - models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
            - models_t5_umt5-xxl-enc-bf16.pth
            - Wan2.1_VAE.pth
            - diffusion_pytorch_model-0000{1-7}-of-00007.safetensors (base models)
        custom_checkpoint_name: Filename of custom DiT checkpoint (e.g., "WoW_video_dit.pt")
        enable_vram_management: Enable VRAM management for memory optimization
        persistent_param_gb: Number of GB to keep persistent in GPU (default 70GB for H800)
    """
    device = f"cuda:{gpu_id}"
    mm = ModelManager(device="cpu")

    checkpoint_folder = Path(checkpoint_folder)
    if not checkpoint_folder.exists():
        raise FileNotFoundError(f"Checkpoint folder does not exist: {checkpoint_folder}")

    # Define model paths
    clip_model_path = checkpoint_folder / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    t5_model_path = checkpoint_folder / "models_t5_umt5-xxl-enc-bf16.pth"
    vae_model_path = checkpoint_folder / "Wan2.1_VAE.pth"
    dit_paths = [
        str(checkpoint_folder / f"diffusion_pytorch_model-0000{i}-of-00007.safetensors")
        for i in range(1, 8)
    ]

    # Load CLIP model
    print("📦 Loading CLIP image encoder...")
    mm.load_models([str(clip_model_path)], torch_dtype=torch.float32)

    # Load DiT, T5, and VAE models
    print("📦 Loading DiT, T5 text encoder and VAE...")
    mm.load_models(
        [dit_paths, str(t5_model_path), str(vae_model_path)],
        torch_dtype=torch.bfloat16,
    )

    # Check for custom checkpoint
    if custom_checkpoint_name:
        custom_checkpoint = checkpoint_folder / custom_checkpoint_name
        if custom_checkpoint.exists():
            print(f"🎯 Loading custom DiT checkpoint: {custom_checkpoint}")
            state_dict = torch.load(str(custom_checkpoint), map_location="cpu")

            dit_model = mm.fetch_model("wan_video_dit")
            if dit_model is not None:
                dit_model.load_state_dict(state_dict, strict=False)
                print("✅ Custom DiT checkpoint loaded successfully")
        else:
            print(f"⚠️  Custom checkpoint not found: {custom_checkpoint}")
            print("   Continuing with base DiT model...")

    # Build pipeline
    pipe = WanVideoPipeline.from_model_manager(mm, torch_dtype=torch.bfloat16, device=device)

    # Configure VRAM management for optimal performance
    if enable_vram_management:
        num_persistent_params = int(persistent_param_gb * 10**9)
        pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_params)
        print(f"✅ VRAM management enabled: {persistent_param_gb}GB persistent params")
    else:
        print("⚠️  VRAM management disabled (may cause OOM on large models)")

    print(f"✅ Pipeline built successfully on {device}")
    return pipe


# Global model variable
pipe = None


def generate_video(prompt, input_file, gpu_id, steps=50, seed=42, tiled=True, num_frames=41):
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
        title="WoW Video Generation Demo",
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
            <h1 class="main-header">🎬 WoW World Generation Studio</h1>
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
                            value=41,
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
    parser = argparse.ArgumentParser(description="WoW Video Generation Demo")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default="dit_models/checkpoints/WoW-1-Wan-14B-600k",
        help="Path to folder containing WAN model files (base models + optional custom checkpoint)"
    )
    parser.add_argument(
        "--custom_checkpoint",
        type=str,
        default="WoW_video_dit.pt",
        help="Filename of custom DiT checkpoint (e.g., 'WoW_video_dit.pt', 'custom_model.pt')"
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
        default=70,
        help="GB of model parameters to keep persistent in GPU memory (default: 70GB for H800, adjust for your GPU)"
    )
    parser.add_argument("--share", action="store_true", help="Create a public link")
    args = parser.parse_args()

    # Handle VRAM management flag logic
    enable_vram = args.enable_vram_management and not args.no_vram_management

    print("=" * 60)
    print(f"🎬 WoW Video Generation Demo")
    print("=" * 60)
    print(f"📍 Checkpoint folder: {args.checkpoint_folder}")
    print(f"📦 Custom checkpoint: {args.custom_checkpoint}")
    print(f"🎮 GPU ID: {args.gpu}")
    print(f"🌐 Port: {args.port}")
    print(f"💾 VRAM Management: {'Enabled' if enable_vram else 'Disabled'}")
    if enable_vram:
        print(f"   Persistent Params: {args.persistent_param_gb}GB")
    print("=" * 60)
    print("⏳ Loading model, please wait...")

    pipe = build_pipeline(
        gpu_id=args.gpu,
        checkpoint_folder=args.checkpoint_folder,
        custom_checkpoint_name=args.custom_checkpoint,
        enable_vram_management=enable_vram,
        persistent_param_gb=args.persistent_param_gb
    )

    print("=" * 60)
    print("✅ Model loaded successfully!")
    print("🚀 Launching Gradio interface...")
    print("=" * 60)

    demo = build_interface()
    demo.launch(server_port=args.port, share=args.share)

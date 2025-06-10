"""
Modal app for FLUX + TRELLIS text-to-image-to-3D generation.
Uses GGUF from  https://huggingface.co/gokaygokay: Flux Game Assets LoRA + Hyper FLUX 8Steps LoRA
We did not use the Flux Game Assets LoRA trigger word as we were happy with the results without it.

We experimented with TRELLIS text-to-3D models (TRELLIS-text-large, TRELLIS-text-xlarge), but we obtained
much better results with the image-to-3D pipeline using Flux.1-dev (with the LoRA weights applied) to generate
the initial image from prompt.

With the current memory management, this container, when staying warm uses less than 30GB of memory
and was tested successfully on A100-40 for many successive inferences without memory leaks.
On a warm container, we obtained 3D asset in less than 30 seconds which is a good result given there
is no optmization in this pipeline yet.

Created for participation in the Hugging Face 🤖 Gradio Agents & MCP Hackathon 2025 🚀
https://huggingface.co/Agents-MCP-Hackathon

Author: castlebbs
"""

import os
import modal
import gc

cuda_version = "12.1.1"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install([
        "git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
        "libxext6", "libxrender-dev", "libgomp1", "libjpeg-dev",
        "build-essential", "ninja-build", "cmake",
    ])
    .env({
        "CUDA_HOME": "/usr/local/cuda-12.1",
        "PYTHONPATH": "/trellis",
        "TORCH_CUDA_ARCH_LIST": "8.0",
        "SPCONV_ALGO": "native",
    })
    .pip_install(
        ["torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0"],
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install([
        "packaging", "wheel", "setuptools", "pillow", "imageio", "imageio-ffmpeg",
        "tqdm", "easydict", "opencv-python-headless", "scipy", "ninja", "rembg",
        "onnxruntime", "trimesh", "open3d", "xatlas", "pyvista", "pymeshfix",
        "igraph", "transformers", "accelerate", "safetensors", "fastapi[standard]",
        "diffusers>=0.30.0", "bitsandbytes", "sentencepiece", "peft", "gguf",
    ])
    .pip_install([
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    ])
    .pip_install(["git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"])
    .pip_install(["kaolin"], extra_options="-f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html")
    .pip_install(["xformers==0.0.27.post2"], extra_options="--index-url https://download.pytorch.org/whl/cu121")
    .pip_install(["spconv-cu121"])
    .pip_install(["git+https://github.com/NVlabs/nvdiffrast.git"])
    .run_commands([
        "mkdir -p /tmp/extensions",
        "git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting",
        "export CXX=g++ && export CC=gcc && pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/",
        "git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /trellis",
    ])
    .workdir("/trellis")
)

app = modal.App("flux-trellis-gguf-3d-pipeline", image=image)

models_volume = modal.Volume.from_name("trellis-models", create_if_missing=True)
torch_hub_volume = modal.Volume.from_name("torch-hub-cache", create_if_missing=True)
rembg_volume = modal.Volume.from_name("rembg-models", create_if_missing=True)

# Global variables to cache pipelines
_flux_pipeline = None
_trellis_pipeline = None
_generation_count = 0

def cleanup_memory():
    """Aggressive memory cleanup without deleting pipelines"""
    global _flux_pipeline, _trellis_pipeline
    
    try:
        import torch
    except ImportError:
        return 
    
    # Clear Flux pipeline caches
    if _flux_pipeline is not None:
        # Clear transformer attention caches
        if hasattr(_flux_pipeline, 'transformer'):
            if hasattr(_flux_pipeline.transformer, 'clear_cache'):
                _flux_pipeline.transformer.clear_cache()
        
        # Clear scheduler state
        if hasattr(_flux_pipeline, 'scheduler'):
            _flux_pipeline.scheduler.timesteps = None
            if hasattr(_flux_pipeline.scheduler, 'sigmas'):
                _flux_pipeline.scheduler.sigmas = None
    
    # Clear TRELLIS pipeline caches
    if _trellis_pipeline is not None:
        # Clear any cached preprocessor states
        if hasattr(_trellis_pipeline, 'image_processor'):
            # Force clear any rembg cached tensors
            try:
                import rembg
                if hasattr(rembg, 'bg_remover'):
                    rembg.bg_remover = None
            except ImportError:
                pass
    
    # Force garbage collection
    gc.collect()
    
    # CUDA cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache() 

def periodic_pipeline_reset():
    """Reset pipelines every 100 generations to prevent memory accumulation"""
    global _flux_pipeline, _trellis_pipeline, _generation_count
    
    _generation_count += 1
    
    if _generation_count % 100 == 0:  
        print(f"Performing periodic pipeline reset after {_generation_count} generations")
        
        # Delete pipelines
        if _flux_pipeline is not None:
            del _flux_pipeline
            _flux_pipeline = None
        
        if _trellis_pipeline is not None:
            del _trellis_pipeline
            _trellis_pipeline = None
        
        # Aggressive cleanup
        gc.collect()
        
        # Import torch for CUDA cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        
        print("Pipeline reset completed")

def get_flux_pipeline():
    """Get or initialize Flux pipeline"""
    global _flux_pipeline
    
    if _flux_pipeline is None:
        import torch
        from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
        from transformers import T5EncoderModel, BitsAndBytesConfig
        
        print("Loading Flux pipeline...")
        dtype = torch.bfloat16
        
        file_url = "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf"
        single_file_base_model = "camenduru/FLUX.1-dev-diffusers"
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            single_file_base_model,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
            quantization_config=quantization_config,
            cache_dir="/models/hf_cache",
        )
        
        # Cache the GGUF file locally first
        gguf_cache_path = "/models/gguf_cache/hyperflux_00001_.q8_0.gguf"
        os.makedirs("/models/gguf_cache", exist_ok=True)
        
        # Only download if not already cached
        if not os.path.exists(gguf_cache_path):
            print("Downloading GGUF file to cache...")
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id="gokaygokay/flux-game",
                filename="hyperflux_00001_.q8_0.gguf",
                cache_dir="/models/hf_cache"
            )
            # Copy to our persistent location
            import shutil
            shutil.copy2(downloaded_path, gguf_cache_path)
            print(f"GGUF file cached at {gguf_cache_path}")
        else:
            print(f"Using cached GGUF file at {gguf_cache_path}")
        
        transformer = FluxTransformer2DModel.from_single_file(
            gguf_cache_path,
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
            config=single_file_base_model,
        )
        
        _flux_pipeline = FluxPipeline.from_pretrained(
            single_file_base_model,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=dtype,
        )
        _flux_pipeline.to("cuda")
        print("Flux pipeline loaded")
    
    return _flux_pipeline

def get_trellis_pipeline(trellis_model_name):
    """Get or initialize TRELLIS pipeline"""
    global _trellis_pipeline
    
    if _trellis_pipeline is None:
        print(f"Loading TRELLIS pipeline: {trellis_model_name}")
        from trellis.pipelines import TrellisImageTo3DPipeline
        
        _trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(trellis_model_name)
        _trellis_pipeline.cuda()
        print("TRELLIS pipeline loaded")
    
    return _trellis_pipeline

@app.function(
    gpu="A100",
    volumes={
        "/models": models_volume,
        "/cache/torch_hub": torch_hub_volume,
        "/cache/rembg": rembg_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=300,
    timeout=3600,
    memory=32768,
    enable_memory_snapshot=True,
)
def text_to_3d(
    text_prompt: str,
    trellis_model_name: str = "JeffreyXiang/TRELLIS-image-large",
    seed: int = 1,
    image_width: int = 1024,
    image_height: int = 1024,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 8,
    ss_guidance_strength: float = 7.5,
    ss_sampling_steps: int = 12,
    slat_guidance_strength: float = 3.0,
    slat_sampling_steps: int = 12,
    mesh_simplify: float = 0.95,
    texture_size: int = 1024,
) -> dict:
    """Generate 3D assets from text prompts using FLUX + TRELLIS pipeline."""
    
    import torch
    import numpy as np
    from PIL import Image
    from trellis.utils import postprocessing_utils
    from io import BytesIO
    
    # Set environment variables
    os.environ.update({
        "SPCONV_ALGO": "native",
        "ATTN_BACKEND": "flash-attn",
        "TORCH_CUDA_ARCH_LIST": "8.0",
        "HF_HOME": "/models/hf_cache",
        "HF_DATASETS_CACHE": "/models/hf_cache",
        "HF_HUB_CACHE": "/models/hf_cache",
        "TORCH_HOME": "/cache/torch_hub",
        "U2NET_HOME": "/cache/rembg",
    })
    
    # Create cache directories
    for path in ["/models/hf_cache", "/cache/torch_hub", "/cache/rembg"]:
        os.makedirs(path, exist_ok=True)
    
    print("Starting pipeline initialization...")
    
    # Check if we need periodic reset
    periodic_pipeline_reset()
    
    # Memory tracking
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"Initial GPU memory: {initial_memory / 1e9:.2f}GB")
    
    try:
        # Get pipelines (will load if not cached)
        flux_pipeline = get_flux_pipeline()
        trellis_pipeline = get_trellis_pipeline(trellis_model_name)
        
        # Generate image with explicit memory management
        print(f"Generating image from prompt: '{text_prompt}'")
        device = "cuda"
        
        with torch.no_grad():
            generator = torch.Generator(device=device).manual_seed(seed)
            
            generated_image = flux_pipeline(
                prompt=text_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=image_width,
                height=image_height,
                generator=generator,
            ).images[0]
        
        # Clear generator and intermediate tensors
        del generator
        cleanup_memory()
        
        print("Image generation completed successfully")
        
        # Preprocess image for TRELLIS
        print("Preprocessing image for 3D generation...")
        with torch.no_grad():
            processed_image = trellis_pipeline.preprocess_image(generated_image)
        
        cleanup_memory()  # Clear rembg intermediate tensors
        print("Image preprocessing completed")
        
        # Generate 3D from image
        print("Generating 3D asset from image...")
        outputs = trellis_pipeline.run(
            processed_image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
        
        cleanup_memory()  # Clear 3D generation intermediate tensors
        print("3D generation completed successfully")
        
        # Prepare result
        result = {
            "text_prompt": text_prompt,
            "seed": seed,
            "trellis_model_name": trellis_model_name,
            "image_generation_params": {
                "width": image_width,
                "height": image_height,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            },
            "3d_generation_params": {
                "ss_guidance_strength": ss_guidance_strength,
                "ss_sampling_steps": ss_sampling_steps,
                "slat_guidance_strength": slat_guidance_strength,
                "slat_sampling_steps": slat_sampling_steps,
            },
        }
        
        # Save generated image
        img_buffer = BytesIO()
        generated_image.save(img_buffer, format="PNG")
        result["generated_image"] = img_buffer.getvalue()
        
        # Generate GLB file
        if outputs.get("gaussian") and outputs.get("mesh"):
            print("Generating GLB file...")
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=mesh_simplify,
                texture_size=texture_size,
            )
            result["glb_file"] = glb.export(file_type="glb")
            print("GLB generation completed successfully")
        else:
            print("Warning: Both gaussian and mesh outputs required for GLB generation")
        
        # Final cleanup
        cleanup_memory()
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        print(f"Final GPU memory: {final_memory / 1e9:.2f}GB")
        print(f"Memory delta: {(final_memory - initial_memory) / 1e6:.1f}MB")
        
        return result
        
    except Exception as e:
        print(f"Error during generation: {e}")
        cleanup_memory()
        raise

@app.local_entrypoint()
def main(
    text_prompt: str = "A isometric 3D dragon with two heads, white background",
    trellis_model_name: str = "JeffreyXiang/TRELLIS-image-large",
    seed: int = 1,
):
    """Local entrypoint for testing the text-to-image-to-3D generation."""
    print(f"Starting text-to-image-to-3D generation...")
    print(f"Prompt: {text_prompt}")
    print(f"TRELLIS Model: {trellis_model_name}")
    print(f"Seed: {seed}")

    result = text_to_3d.remote(
        text_prompt=text_prompt,
        trellis_model_name=trellis_model_name,
        seed=seed,
    )

    print(f"Generation completed!")
    print(f"Result keys: {list(result.keys())}")

    import os
    output_dir = "modal_outputs"
    os.makedirs(output_dir, exist_ok=True)

    if "generated_image" in result:
        with open(os.path.join(output_dir, "generated_image.png"), "wb") as f:
            f.write(result["generated_image"])
        print(f"Saved: {output_dir}/generated_image.png")

    if "glb_file" in result:
        with open(os.path.join(output_dir, "model.glb"), "wb") as f:
            f.write(result["glb_file"])
        print(f"Saved: {output_dir}/model.glb")

    return result

if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "A isometric 3D dragon with two heads, white background"
    model = sys.argv[2] if len(sys.argv) > 2 else "JeffreyXiang/TRELLIS-image-large"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    with app.run():
        main(prompt, model, seed)
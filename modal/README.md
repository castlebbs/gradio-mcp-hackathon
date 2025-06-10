
## Modal remote function
This code deploys the text_to_3d remote function to Modal.

### FLUX Pipeline Models

Here is a description of the different models used in this pipeline.

1. **T5 Text Encoder** 
   - Type: Transformer-based text encoder
   - Function: Converts text prompts to embeddings
   - Optimization: 8-bit quantization via BitsAndBytesConfig

2. **FLUX Transformer2D Model**
   - Type: Diffusion transformer for image generation
   - Function: Core image generation from text embeddings
   - Optimization: GGUF q8_0 quantization (from https://huggingface.co/gokaygokay)
   - LoRA Models Applied:
     - **Hyper FLUX 8Steps LoRA**: Reduces inference steps from ~20-50 to 8
     - **gokaygokay's Flux Game Assets LoRA**: Improves 3D asset generation quality

### TRELLIS Pipeline Models:
3. **TRELLIS-image-large**
   - Type: Multi-stage 3D generation model
   - Function: Converts 2D images to 3D representations
   - Components:
     - **Sparse Structure Sampler**: Initial 3D structure generation
     - **SLAT (Structured Latent) Sampler**: 3D refinement and detail enhancement

### Supporting Models:
4. **U2NET (via rembg)**
   - Type: Salient object detection model
   - Function: Background removal for clean 3D generation
   - Used by: rembg library in TRELLIS preprocessing

### Post-processing Components:
5. **Mesh Simplification Algorithm**
   - Function: Reduces polygon count (95% reduction default)
6. **Texture Generator**
   - Function: Creates 1024x1024 textures for 3D models
7. **GLB Exporter**
   - Function: Combines gaussian splatting + mesh into GLB format

### Pipeline Sequence
![pipeline-sequence](https://github.com/user-attachments/assets/5268a95f-8b6a-48e3-8bcc-86028b69ab46)




import json
import re
import anthropic
import gradio as gr
import modal
import base64
import os
import tempfile

# Modal function to run the 3D generation pipeline
text_to_3d = modal.Function.from_name("flux-trellis-gguf-3d-pipeline", "text_to_3d")


def generate_3d_prompts(player_bio: str, num_assets: int = 10) -> list:
    """
    Analyze player bio using Claude Sonnet to generate 3D asset prompts.

    Args:
        player_bio (str): The player's biographical information
        num_assets (int): Number of assets to generate (default: 10)

    Returns:
        list: List of 3D generation prompts
    """
    client = anthropic.Anthropic()

    system_prompt = f"""You are an expert 3D scene designer for video games. Your task is to analyze a player's biographical information and identify specific 3D assets that would create a comfortable, personalized environment for them.

Based on the player's bio, generate a list of exactly {num_assets} specific 3D asset prompts that would compose a scene tailored to their interests, personality, and background.

Objects should be deeply personal and couldn't exist in any generic asset library. You should combine multiple elements into a single prompt if they are closely related. For instance, if the player loves both coffee and gaming, you might create a prompt for a "gaming desk with a custom coffee station."
Prefer 3d assets, which combine multiple elements instead of single objects like a piece of furniture or a decoration.
Each prompt should include: 3d isomorphic, white background. This is for 3D game asset generation.
Format your response as a simple JSON array of strings, where each string is a detailed prompt for 3D generation. Focus on objects like furniture, decorations, equipment, or environmental elements.

Follow this format:
<output>
```json
[
    "prompt 1",
    "prompt 2",
    "prompt 3",
]
```
</output>
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Player bio: {player_bio}"}],
    )

    prompts_text = response.content[0].text

    # Extract JSON from the response text using regex
    try:
        # Find JSON array between ```json ``` markers
        json_pattern = r"```json\s*(\[.*?\])\s*```"
        match = re.search(json_pattern, prompts_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        else:
            # Fallback: try to find any JSON array
            array_pattern = r"(\[.*?\])"
            match = re.search(array_pattern, prompts_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            else:
                return [prompts_text]
    except (json.JSONDecodeError, ValueError):
        # Fallback: return the raw text if JSON parsing fails
        return [prompts_text]


def generate_3d_assets(player_bio: str, num_assets: int = 10) -> str:
    """
    Generate 3D assets based on a player's bio for video game scene composition.

    Args:
        player_bio (str): The player's biographical information
        num_assets (int): Number of assets to generate

    Returns:
        str: JSON string containing prompts and GLB file data
    """
    try:
        # Analyze bio with Claude to get 3D prompts
        prompts = generate_3d_prompts(player_bio, num_assets)

        # Generate 3D assets using Modal function
        modal_results = text_to_3d.map(prompts)

        # Process results and prepare GLB data for API/MCP consumption
        generated_assets = []
        for i, result in enumerate(modal_results):
            if "glb_file" in result:
                generated_assets.append(
                    {
                        "prompt": prompts[i],
                        "glb_data": base64.b64encode(result["glb_file"]).decode(
                            "utf-8"
                        ),
                        "size_bytes": len(result["glb_file"]),
                        "asset_id": f"asset_{i+1}",
                    }
                )

        result_dict = {
            "assets": generated_assets,
            "total_assets": len(generated_assets),
        }

        return json.dumps(result_dict)

    except Exception as e:
        error_dict = {"error": str(e), "prompts": [], "assets": [], "total_assets": 0}
        return json.dumps(error_dict)


def generate_3d_assets_with_display(player_bio: str, num_assets: int = 10) -> tuple:
    """
    Generate 3D assets and prepare them for both JSON output and 3D display.

    Returns:
        tuple: (json_output, model_paths_dict, num_assets)
    """
    result_json = generate_3d_assets(player_bio, num_assets)
    result = json.loads(result_json)

    if "error" in result:
        return result_json, {}, num_assets

    # Save GLB files to temporary directory for display
    temp_dir = tempfile.mkdtemp()
    model_paths = {}

    for i, asset in enumerate(result.get("assets", [])):
        if "glb_data" in asset:
            # Decode base64 back to bytes and save to file
            glb_bytes = base64.b64decode(asset["glb_data"])
            model_path = os.path.join(temp_dir, f"model_{i+1}.glb")
            with open(model_path, "wb") as f:
                f.write(glb_bytes)
            model_paths[f"model_{i+1}"] = model_path

    return result_json, model_paths, num_assets

# Create a clean Gradio interface using Blocks
with gr.Blocks(theme=gr.themes.Monochrome(), title="3D Scene Asset Generator") as demo:

    gr.Markdown("# 🎮 3D Game Environment Builder  🏗️")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown(
                """
                Transform player personalities into immersive 3D game environments! Create fast personalized 3D environments to a player, by using information from 
                their bio. This tool will craft personalized 3D assets that reflect each player's unique interests, hobbies, and lifestyle, allowing you to build unique scenes for video games.
                
                
                This space is used as a MCP Server, example usage:
                ```bash
                mcptools call generate_3d_assets --params '{"player_bio":"Elena is a music producer who [...]"}' https://{gradio-url}:7860/gradio_api/mcp/sse
                ```
                This returns a JSON with the generated 3D assets in GLB format, along with their description.
                """
            )
        with gr.Column(scale=1):
            gr.HTML(
                """
                <iframe width="100%" height="315" src="https://www.youtube.com/embed/09Dk9OL65bc" 
                        frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen></iframe>
                """
            )
    
    gr.Image("images/pipeline.png", label="🎯 Pipeline Overview", show_label=True, height=200)

    gr.Markdown(
        """
        <div style="background-color: #e7f3ff; border-left: 4px solid #2196F3; padding: 16px; margin: 16px 0; border-radius: 4px;">
            <h4 style="color: #1976D2; margin-top: 0; margin-bottom: 8px;">ℹ️ Testing Information</h4>
            <p style="margin: 0; color: #333;">
                <strong>Note:</strong> This space cannot be directly tested as we didn't want users to enter their own API keys for the various providers used in the pipeline.
            </p>
            <p style="margin: 8px 0 0 0; color: #333;">
                However, we provide all the source code and the instructions to build your own and test it at: 
                <a href="https://github.com/castlebbs/gradio-mcp-hackathon/" target="_blank" style="color: #1976D2; text-decoration: none; font-weight: bold;">🔗 GitHub Repository</a>
            </p>
        </div>
        """
    )    

    with gr.Row():
        with gr.Column(scale=3):
            bio_input = gr.Textbox(
                label="📝 Player Bio",
                placeholder=(
                    "Tell us about the player's interests, hobbies, personality, "
                    "and lifestyle...\n\n"
                    "Example: Sarah is an avid reader who loves fantasy novels and "
                    "cozy spaces. She enjoys knitting while listening to classical "
                    "music and has a collection of houseplants. Her ideal environment "
                    "would be warm and inviting with lots of natural elements."
                ),
                lines=8,
                max_lines=10,
            )
        
        with gr.Column(scale=1):
            num_assets_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="🎯 Number of 3D Assets to Generate.",
                info="Select how many 3D assets you want to generate (1-10).\n\nA"
                      "Assets generation run in parallel in dedicated pipelines."
                      "Each pipeline runs on a separate Modal Container with A100-40GB"
                      "attached to them",
            )

    with gr.Row():
        generate_btn = gr.Button(
            "✨ Generate 3D Assets", variant="primary", size="lg"
        )

    with gr.Row():
        with gr.Column(scale=1):
            output_json = gr.JSON(label="🎯 Generated 3D Assets Results")
        with gr.Column(scale=1):
            # Create tabs that will be shown/hidden based on num_assets
            model_tabs = gr.Tabs()
            with model_tabs:
                model_components = []
                for i in range(1, 11):
                    with gr.Tab(f"Model {i}", visible=(i <= 5)) as tab:
                        model_3d = gr.Model3D(label=f"🎨 3D Model {i}", height=400)
                        model_components.append((tab, model_3d))

    # Add examples
    gr.Examples(
        examples=[
            [
                "Alice is a nature lover who enjoys reading fantasy novels. She has "
                "a collection of vintage books and loves to spend time in her cozy "
                "garden with her cat, Whiskers. Alice also enjoys painting landscapes "
                "and has a small easel set up in her living room.",
                5,
            ],
            [
                "Marcus is a tech enthusiast and gaming streamer who loves mechanical "
                "keyboards and collecting vintage arcade games. He's also a coffee "
                "connoisseur who roasts his own beans and enjoys late-night coding "
                "sessions.",
                7,
            ],
            [
                "Elena is a music producer who plays multiple instruments including "
                "piano and guitar. She has a home studio filled with vintage "
                "synthesizers and loves vinyl records. She also practices yoga and "
                "meditation in her spare time.",
                6,
            ],
        ],
        inputs=[bio_input, num_assets_slider],
        label="💡 Try these example biographies:",
    )

    def update_models(player_bio: str, num_assets: int):
        """Handle model updates for all tabs"""
        json_result, model_paths, _ = generate_3d_assets_with_display(
            player_bio, num_assets
        )

        # Prepare outputs for all components
        tab_updates = []
        model_updates = []

        for i in range(1, 11):
            # Update tab visibility
            tab_updates.append(gr.update(visible=(i <= num_assets)))

            # Update model content
            model_key = f"model_{i}"
            if i <= num_assets and model_key in model_paths:
                model_updates.append(model_paths[model_key])
            else:
                model_updates.append(None)

        return [json_result] + tab_updates + model_updates

    # Update tab visibility when slider changes
    def update_tab_visibility(num_assets: int):
        tab_updates = []
        for i in range(1, 11):
            tab_updates.append(gr.update(visible=(i <= num_assets)))
        return tab_updates

    num_assets_slider.change(
        fn=update_tab_visibility,
        inputs=[num_assets_slider],
        outputs=[tab for tab, _ in model_components],
        api_name=False,
    )

    generate_btn.click(
        fn=update_models,
        inputs=[bio_input, num_assets_slider],
        outputs=[output_json]
        + [tab for tab, _ in model_components]
        + [model for _, model in model_components],
        show_progress=True,
        api_name=False,
    )

    # API components
    api_input = gr.Textbox(visible=False)
    api_num_assets = gr.Number(visible=False, value=10)
    api_output = gr.Textbox(visible=False)
    api_btn = gr.Button(visible=False)

    api_btn.click(
        fn=generate_3d_assets,
        inputs=[api_input, api_num_assets],
        outputs=api_output,
        api_name="generate_3d_assets",
    )

# Launch the Gradio web interface and MCP server
if __name__ == "__main__":
    demo.launch(mcp_server=True, share=False)

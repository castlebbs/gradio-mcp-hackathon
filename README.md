# 🎮 3D Scene Asset Generator
### Our participation to the 2025 Gradio Agent MCP Hackathon

> Transform player biographies into personalized 3D environments using LLM-powered analysis and 3D asset generation models pipelines.

## 🌟 Project Overview

This hackathon project creates a 3D scene generator that analyzes player biographies and automatically generates personalized 3D environments. By combining the power of LLM analysis with generation models (FLUX + Trellis), we create unique, contextual 3D assets that reflect each player's personality, interests, and background.

## ✨ Key Features

- **🤖 AI-Powered Analysis**: LLM analyzes player biographies to understand personality and interests
- **🎨 3D Generation**: FLUX + Trellis pipeline generates high-quality, contextual 3D assets
- **🌐 Interactive Web Interface**: Gradio interface with real-time generation and examples
- **🎮 3D Game Integration**: Godot game client that connects to MCP server for immersive 3D environment visualization
- **🔧 MCP Integration**: Supports Model Context Protocol for enhanced interactions
- **⚡ Optimized Pipeline**: Uses GGUF quantization and LoRA models for fast, efficient generation
- **📱 User-Friendly**: Simple input → AI analysis → 3D asset generation → game environment workflow

## 🏗️ Architecture

![MCP Hackaton-3](https://github.com/user-attachments/assets/b135ce3a-43d6-4f1d-855b-92af4bce65c0)


Thank you to gokaygokay for the GGUF

### Technology Stack

- **Frontend**: Gradio with custom CSS styling
- **Game Client**: Godot Engine 4.4 for 3D environment visualization
- **AI Analysis**: Anthropic Claude Sonnet 4
- **3D Generation**: FLUX + Trellis on Modal
- **MCP Protocol**: Model Context Protocol for client-server communication
- **Output Format**: GLB (3D models compatible with most engines)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Anthropic API key
- Modal account
- Godot Engine 4.4+ (for game client)
- mcptools CLI (for MCP communication)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/castlebbs/gradio-mcp-hackathon.git
   cd gradio-mcp-hackathon
   ```

2. **Set up the Gradio application**
   ```bash
   cd gradio
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

4. **Set up Modal**
   ```bash
   modal setup
   ```

5. **Deploy the Modal function**
   ```bash
   cd ../modal
   modal deploy flux-trellis-GGUF-text-to-3d.py
   ```

6. **Run the application**
   ```bash
   cd ../gradio
   python app.py
   ```

7. **Set up the Godot game client**

- Install mcptools for MCP communication. Check your OS install instruction on: https://github.com/f/mcptools/blob/master/README.md
- Open the Godot project
- In Godot Engine, open: godot/project.godot
- Run the game scene to start the 3D environment

## 💡 Usage Example

### Web Interface
**Input Biography:**
> "Marcus is a tech enthusiast and gaming streamer who loves mechanical keyboards and collecting vintage arcade games. He's also a coffee connoisseur who roasts his own beans and enjoys late-night coding sessions."

**Generated 3D Assets:**
- Vintage arcade cabinet with classic game artwork
- Premium mechanical keyboard with RGB backlighting  
- Professional coffee roasting station with custom setup
- Gaming chair with LED accents and streaming equipment
- Retro-futuristic desk lamp with adjustable lighting

### Godot Game Client
The Godot game provides an immersive 3D environment where:
1. **Player Input**: Enter your biography through the in-game UI
2. **MCP Communication**: Game connects to the Gradio MCP server via mcptools
3. **Real-time Generation**: 3D assets are generated and sent back to the game
4. **Environment Building**: Assets are automatically placed in the 3D scene
5. **Interactive Exploration**: Walk around and explore your personalized environment

## 📁 Project Structure

```
gradio-mcp-hackathon/
├── gradio/                    # Main Gradio application
│   ├── app.py                # Core application logic
│   ├── requirements.txt      # Python dependencies
│   ├── README.md            # Detailed app documentation
│   └── images/              # UI assets and examples
├── modal/                    # Modal cloud functions
│   ├── flux-trellis-GGUF-text-to-3d.py  # 3D generation pipeline
│   └── README.md            # Modal setup documentation
├── godot/                    # Godot game client
│   ├── project.godot        # Godot project configuration
│   ├── mcp.sh              # MCP communication script (Unix/macOS)
│   ├── mcp.bat             # MCP communication script (Windows)
│   ├── scenes/             # Game scenes (main, player, UI)
│   ├── scripts/            # GDScript files for game logic
│   │   ├── main.gd         # Main scene controller
│   │   ├── ui.gd           # User interface logic
│   │   ├── mcp.gd          # MCP client communication
│   │   └── 3Dgeneration.gd # 3D asset handling and placement
│   ├── assets/             # Generated 3D assets storage
│   └── models/             # Base 3D models and textures
├── LICENSE                   # MIT License
└── README.md                # This file
```

## 🔧 Technical Details

### AI Pipeline
- **Text Analysis**: Claude Sonnet processes biographical text to extract personality traits and interests
- **Prompt Generation**: AI creates detailed, contextual prompts for 3D asset generation
- **Asset Creation**: FLUX + Trellis pipeline generates high-quality 3D models

### Optimizations
- **GGUF Quantization**: Reduces model size while maintaining quality
- **LoRA Models**: Hyper FLUX 8Steps for faster inference, Game Assets LoRA for better 3D results
- **Modal Scaling**: Automatic scaling for concurrent requests

## 🏆 Hackathon Team

- castlebbs@ - Gradio, Modal
- stargarnet@ - Godot
- zinkenite@ - 3D work

Built with ❤️ for the 2025 Gradio Agent MCP Hackathon

## Links

- https://huggingface.co/black-forest-labs/FLUX.1-dev
- https://huggingface.co/microsoft/TRELLIS-image-large
- https://huggingface.co/spaces/gokaygokay/Flux-TRELLIS
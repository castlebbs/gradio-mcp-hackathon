# 🎮 3D Scene Asset Generator
### Our participation to the 2025 Gradio Agent MCP Hackathon

> Transform player biographies into personalized 3D environments using AI-powered analysis and 3D asset generation models.

## 🌟 Project Overview

This hackathon project creates a 3D scene generator that analyzes player biographies and automatically generates personalized 3D environments. By combining the power of LLM analysis with generation models (FLUX + Trellis), we create unique, contextual 3D assets that reflect each player's personality, interests, and background.

## ✨ Key Features

- **🤖 AI-Powered Analysis**: LLM analyzes player biographies to understand personality and interests
- **🎨 3D Generation**: FLUX + Trellis pipeline generates high-quality, contextual 3D assets
- **🌐 Interactive Web Interface**: Gradio interface with real-time generation and examples
- **🔧 MCP Integration**: Supports Model Context Protocol for enhanced interactions
- **⚡ Optimized Pipeline**: Uses GGUF quantization and LoRA models for fast, efficient generation
- **📱 User-Friendly**: Simple input → AI analysis → 3D asset generation workflow

## 🏗️ Architecture

![MCP Hackaton-3](https://github.com/user-attachments/assets/b135ce3a-43d6-4f1d-855b-92af4bce65c0)


Thank you to gokaygokay for the GGUF

### Technology Stack

- **Frontend**: Gradio with custom CSS styling
- **AI Analysis**: Anthropic Claude Sonnet 4
- **3D Generation**: FLUX + Trellis on Modal
- **Output Format**: GLB (3D models compatible with most engines)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Anthropic API key
- Modal account

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

## 💡 Usage Example

**Input Biography:**
> "Marcus is a tech enthusiast and gaming streamer who loves mechanical keyboards and collecting vintage arcade games. He's also a coffee connoisseur who roasts his own beans and enjoys late-night coding sessions."

**Generated 3D Assets:**
- Vintage arcade cabinet with classic game artwork
- Premium mechanical keyboard with RGB backlighting  
- Professional coffee roasting station with custom setup
- Gaming chair with LED accents and streaming equipment
- Retro-futuristic desk lamp with adjustable lighting

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

## 🎯 Hackathon Goals

This project demonstrates:
- ✅ **MCP Integration**: Seamless AI-to-AI communication
- ✅ **Gradio Excellence**: Beautiful, functional web interface  
- ✅ **AI Innovation**: Novel application of LLMs for 3D content creation
- ✅ **Technical Excellence**: Optimized pipeline with modern ML techniques
- ✅ **User Experience**: Simple workflow hiding complex AI operations

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Hackathon Team

Built with ❤️ for the 2025 Gradio Agent MCP Hackathon

# 3D Scene Asset Generator

A Gradio-based application that transforms player biographies into personalized 3D environments using AI-powered analysis and 3D asset generation.

## Features

- **AI-Powered Analysis**: Uses Claude Sonnet to analyze player biographies and generate contextual 3D asset prompts
- **3D Asset Generation**: Integrates with Trellis API and Modal for high-quality 3D model generation
- **Interactive Web Interface**: Beautiful Gradio interface with examples and real-time generation
- **MCP Server**: Supports Model Context Protocol for enhanced AI interactions

## Requirements

- Python 3.8+
- Anthropic API key
- Modal account and setup
- Required Python packages (see installation)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gradio-mcp
```

2. Install dependencies:
```bash
pip install gradio anthropic modal requests
```

3. Set up your API keys:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

4. Configure Modal:
```bash
modal setup
```

## Usage

Run the application:
```bash
python app.py
```

The application will launch a Gradio interface where you can:
1. Enter a player biography
2. Generate personalized 3D asset prompts
3. View the generated prompts and download 3D models

## Example

Input a player biography like:
> "Marcus is a tech enthusiast and gaming streamer who loves mechanical keyboards and collecting vintage arcade games. He's also a coffee connoisseur who roasts his own beans and enjoys late-night coding sessions."

The system will generate contextual prompts for 3D assets like:
- Vintage arcade cabinet with classic game artwork
- Premium mechanical keyboard with RGB backlighting
- Professional coffee roasting machine
- Gaming chair with LED accents
- Retro-futuristic desk lamp

## Architecture

- **Frontend**: Gradio web interface with custom CSS styling
- **AI Analysis**: Claude Sonnet for bio analysis and prompt generation
- **3D Generation**: Modal functions calling Trellis API
- **Output**: GLB format 3D models saved locally

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

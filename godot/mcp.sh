#!/bin/zsh

# Location of the mcptools command
MCPTOOLS_CMD="/opt/homebrew/bin/mcptools"

# MCP server URL
MCP_SERVER_URL="http://127.0.0.1:7860/gradio_api/mcp/sse"

# Check if bio parameter is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <player_bio> [num_assets]"
    exit 1
fi

# Set default number of assets if not provided
NUM_ASSETS=2
if [ -n "$2" ]; then
    NUM_ASSETS="$2"
fi

# Construct JSON with the bio and num_assets parameters
PLAYER_BIO="$1"
JSON_PARAMS="{\"player_bio\":\"$PLAYER_BIO\",\"num_assets\":$NUM_ASSETS}"

# Execute the mcptools command
"$MCPTOOLS_CMD" call 3dgen_generate_3d_assets --params "$JSON_PARAMS" "$MCP_SERVER_URL"

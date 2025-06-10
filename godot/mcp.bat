@echo off

REM Configuration
set "MCPTOOLS_CMD=mcptools"
set "MCP_SERVER_URL=http://127.0.0.1:7860/gradio_api/mcp/sse"

REM Check if bio parameter is provided
if "%~1"=="" (
    echo Usage: %0 ^<player_bio^> [num_assets]
    exit /b 1
)

REM Set default number of assets if not provided
set "NUM_ASSETS=5"
if not "%~2"=="" (
    set "NUM_ASSETS=%~2"
)

REM Construct JSON with the bio and num_assets parameters
set "PLAYER_BIO=%~1"
set "JSON_PARAMS={\"player_bio\":\"%PLAYER_BIO%\",\"num_assets\":%NUM_ASSETS%}"

REM Execute the mcptools command
%MCPTOOLS_CMD% call 3dgen_generate_3d_assets --params "%JSON_PARAMS%" "%MCP_SERVER_URL%"
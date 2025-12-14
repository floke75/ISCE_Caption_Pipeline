#!/bin/bash
set -e

echo "Starting setup..."

# Install system dependencies
if command -v apt-get >/dev/null; then
    echo "Installing system packages..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg git build-essential
else
    echo "Warning: apt-get not found. Skipping system package installation."
fi

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install Core dependencies explicitly to ensure order and success
echo "Installing core Python dependencies..."
# Core + Torch + Spacy + Web/Test
pip install pyyaml pandas numpy rapidfuzz tqdm pysrt fastapi pydantic "uvicorn[standard]" pytest httpx ffmpeg-python ruff
pip install torch
pip install "spacy>=3.7,<4.0"

# Download SpaCy model
echo "Downloading SpaCy model..."
python -m spacy download sv_core_news_lg

# Try installing full requirements (including whisperx)
echo "Attempting to install full requirements.txt..."
pip install -r requirements.txt || echo "Warning: Full requirements install failed (likely whisperx/pyannote). Core usage should still work."

echo "Setup script completed successfully."

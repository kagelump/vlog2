#!/bin/bash
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Model directory name
MODEL_DIR_NAME="models--mlx-community--Qwen3-VL-8B-Instruct-8bit"
HF_CACHE_DIR="$HOME/.cache/huggingface/hub"

echo -e "${GREEN}TVAS Model Installer${NC}"
echo "This script will copy the AI model to your local Hugging Face cache."

# Determine source directory
# We assume this script is located in the root of the SD card or distribution folder
# and the 'huggingface' folder is in the same directory or parent directory.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SOURCE_MODEL_PATH=""

# Check if running from the SD card root where 'huggingface' folder exists
if [ -d "$SCRIPT_DIR/huggingface/hub/$MODEL_DIR_NAME" ]; then
    SOURCE_MODEL_PATH="$SCRIPT_DIR/huggingface/hub/$MODEL_DIR_NAME"
elif [ -d "$SCRIPT_DIR/../huggingface/hub/$MODEL_DIR_NAME" ]; then
    SOURCE_MODEL_PATH="$SCRIPT_DIR/../huggingface/hub/$MODEL_DIR_NAME"
else
    # Try to find it in common mount locations if not found relative to script
    POSSIBLE_MOUNTS=$(find /Volumes -maxdepth 2 -name "$MODEL_DIR_NAME" 2>/dev/null || true)
    if [ -n "$POSSIBLE_MOUNTS" ]; then
        SOURCE_MODEL_PATH=$(echo "$POSSIBLE_MOUNTS" | head -n 1)
    fi
fi

if [ -z "$SOURCE_MODEL_PATH" ]; then
    echo -e "${RED}Error: Could not find the model directory '$MODEL_DIR_NAME'.${NC}"
    echo "Please ensure the 'huggingface' folder is present on the SD card or in the same directory as this script."
    exit 1
fi

echo -e "Found model at: ${YELLOW}$SOURCE_MODEL_PATH${NC}"

# Create destination directory
mkdir -p "$HF_CACHE_DIR"

DEST_MODEL_PATH="$HF_CACHE_DIR/$MODEL_DIR_NAME"

if [ -d "$DEST_MODEL_PATH" ]; then
    echo -e "${YELLOW}Model already exists at $DEST_MODEL_PATH${NC}"
    read -p "Do you want to overwrite it? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping installation."
        exit 0
    fi
    rm -rf "$DEST_MODEL_PATH"
fi

echo -e "${GREEN}Copying model files... This may take a while.${NC}"
cp -R "$SOURCE_MODEL_PATH" "$HF_CACHE_DIR/"

echo -e "${GREEN}Model installed successfully!${NC}"
echo "Location: $DEST_MODEL_PATH"

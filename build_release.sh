#!/bin/bash
set -e

# Define colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting build process for TVAS...${NC}"

# Create a build directory
BUILD_DIR="build_artifacts"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Create a virtual environment for building
echo -e "${GREEN}Creating virtual environment...${NC}"
python3 -m venv "$BUILD_DIR/venv"
source "$BUILD_DIR/venv/bin/activate"

# Install build dependencies and the project itself
echo -e "${GREEN}Installing dependencies...${NC}"
pip install --upgrade pip
pip install pyinstaller
pip install .

# Find mlx.metallib path
MLX_METALLIB=$(python -c "import os, mlx; print(os.path.join(list(mlx.__path__)[0], 'lib', 'mlx.metallib'))")
echo "Found mlx.metallib at: $MLX_METALLIB"

# Build TVAS (Main App)
echo -e "${GREEN}Building TVAS executable...${NC}"
pyinstaller --noconfirm --clean \
    --name tvas \
    --add-data "src/shared/prompts/*.txt:shared/prompts" \
    --add-data "$MLX_METALLIB:." \
    --collect-all mlx \
    --collect-all mlx_vlm \
    --collect-all toga \
    --hidden-import=tvas \
    --hidden-import=tvas.main \
    --hidden-import=shared \
    src/tvas/main.py

# Find HuggingFace cache directory
HF_CACHE=$(python -c "from huggingface_hub import snapshot_download; import os; print(os.path. dirname(snapshot_download('mlx-community/Qwen3-VL-8B-Instruct-8bit', cache_dir=None)))")
echo "Found HuggingFace cache at: $HF_CACHE"

# Generate ICNS file
echo -e "${GREEN}Generating App Icon...${NC}"
mkdir -p "$BUILD_DIR/icons.iconset"
# Convert WebP to PNG using Pillow
python -c "from PIL import Image; Image.open('assets/tprslogo.webp').save('$BUILD_DIR/logo.png')"

# Create various sizes for the iconset
sips -z 16 16     "$BUILD_DIR/logo.png" --out "$BUILD_DIR/icons.iconset/icon_16x16.png" > /dev/null
sips -z 32 32     "$BUILD_DIR/logo.png" --out "$BUILD_DIR/icons.iconset/icon_16x16@2x.png" > /dev/null
sips -z 32 32     "$BUILD_DIR/logo.png" --out "$BUILD_DIR/icons.iconset/icon_32x32.png" > /dev/null
sips -z 64 64     "$BUILD_DIR/logo.png" --out "$BUILD_DIR/icons.iconset/icon_32x32@2x.png" > /dev/null
sips -z 128 128   "$BUILD_DIR/logo.png" --out "$BUILD_DIR/icons.iconset/icon_128x128.png" > /dev/null
sips -z 256 256   "$BUILD_DIR/logo.png" --out "$BUILD_DIR/icons.iconset/icon_128x128@2x.png" > /dev/null
sips -z 256 256   "$BUILD_DIR/logo.png" --out "$BUILD_DIR/icons.iconset/icon_256x256.png" > /dev/null
sips -z 512 512   "$BUILD_DIR/logo.png" --out "$BUILD_DIR/icons.iconset/icon_256x256@2x.png" > /dev/null
sips -z 512 512   "$BUILD_DIR/logo.png" --out "$BUILD_DIR/icons.iconset/icon_512x512.png" > /dev/null
sips -z 1024 1024 "$BUILD_DIR/logo.png" --out "$BUILD_DIR/icons.iconset/icon_512x512@2x.png" > /dev/null

# Convert iconset to icns
iconutil -c icns "$BUILD_DIR/icons.iconset" -o "$BUILD_DIR/tprs.icns"

# Get absolute path for icon
ICON_PATH="$(pwd)/$BUILD_DIR/tprs.icns"

# Build TPRS (Photo Rating CLI)
echo -e "${GREEN}Building TPRS executable...${NC}"
pyinstaller --noconfirm --clean \
    --name tprs \
    --windowed \
    --icon "$ICON_PATH" \
    --osx-bundle-identifier=com.kagelump.tprs \
    --add-data "src/shared/prompts/*.txt:shared/prompts" \
    --add-data "$MLX_METALLIB:." \
    --collect-all mlx \
    --collect-all mlx_vlm \
    --collect-all toga \
    --hidden-import=tprs \
    --hidden-import=tprs.cli \
    --hidden-import=shared \
    src/tprs/cli.py

# Create the distribution folder
DIST_NAME="tvas_release"
rm -rf "$DIST_NAME"
mkdir -p "$DIST_NAME"

# Copy executables to distribution folder
cp -r dist/tvas "$DIST_NAME/"
cp -r dist/tprs.app "$DIST_NAME/"

# Force Finder to refresh the icon
touch "$DIST_NAME/tprs.app"

# Copy README
cp README.md "$DIST_NAME/"

# Create a helper script to run them easily (optional, but helpful)
cat << EOF > "$DIST_NAME/install.sh"
#!/bin/bash
# Simple install script to add to path or just run
echo "You can run the tools directly from this folder:"
echo "  ./tvas/tvas"
echo "  open ./tprs.app"
echo ""
echo "To install to /usr/local/bin (requires sudo):"
echo "  sudo ln -sf \$(pwd)/tvas/tvas /usr/local/bin/tvas"
echo "  sudo ln -sf \$(pwd)/tprs.app/Contents/MacOS/tprs /usr/local/bin/tprs"
EOF
chmod +x "$DIST_NAME/install.sh"

# Zip the distribution
ZIP_NAME="tvas_mac_release.zip"
echo -e "${GREEN}Zipping release to $ZIP_NAME...${NC}"
zip -r "$ZIP_NAME" "$DIST_NAME"

echo -e "${GREEN}Build complete!${NC}"
echo -e "You can now copy ${GREEN}$ZIP_NAME${NC} to another Mac."

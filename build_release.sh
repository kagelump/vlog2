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
    --add-data "src/tvas/prompts/*.txt:tvas/prompts" \
    --add-data "$MLX_METALLIB:." \
    --collect-all mlx \
    --collect-all mlx_vlm \
    --collect-all toga \
    --hidden-import=tvas \
    --hidden-import=tvas.main \
    src/tvas/main.py

# Build TPRS (Photo Rating CLI)
echo -e "${GREEN}Building TPRS executable...${NC}"
pyinstaller --noconfirm --clean \
    --name tprs \
    --add-data "src/tvas/prompts/*.txt:tvas/prompts" \
    --add-data "$MLX_METALLIB:." \
    --collect-all mlx \
    --collect-all mlx_vlm \
    --collect-all toga \
    --hidden-import=tvas \
    --hidden-import=tvas.tprs_cli \
    src/tvas/tprs_cli.py

# Create the distribution folder
DIST_NAME="tvas_release"
rm -rf "$DIST_NAME"
mkdir -p "$DIST_NAME"

# Copy executables to distribution folder
cp -r dist/tvas "$DIST_NAME/"
cp -r dist/tprs "$DIST_NAME/"

# Copy README
cp README.md "$DIST_NAME/"

# Create a helper script to run them easily (optional, but helpful)
cat << EOF > "$DIST_NAME/install.sh"
#!/bin/bash
# Simple install script to add to path or just run
echo "You can run the tools directly from this folder:"
echo "  ./tvas/tvas"
echo "  ./tprs/tprs"
echo ""
echo "To install to /usr/local/bin (requires sudo):"
echo "  sudo ln -sf \$(pwd)/tvas/tvas /usr/local/bin/tvas"
echo "  sudo ln -sf \$(pwd)/tprs/tprs /usr/local/bin/tprs"
EOF
chmod +x "$DIST_NAME/install.sh"

# Zip the distribution
ZIP_NAME="tvas_mac_release.zip"
echo -e "${GREEN}Zipping release to $ZIP_NAME...${NC}"
zip -r "$ZIP_NAME" "$DIST_NAME"

echo -e "${GREEN}Build complete!${NC}"
echo -e "You can now copy ${GREEN}$ZIP_NAME${NC} to another Mac."

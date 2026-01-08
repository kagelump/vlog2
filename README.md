# Vlog tools

A set of tools to help with travel videos and photos, including:

* TVAS - Process videos for DaVinci Resolve
* TPRS - Process photos for Lightroom or DxO PhotoLab

## TPRS Quickstart (Recommended with LM Studio)

For the best performance, we recommend using LM Studio to host the vision model locally.
For simplicity, we recommend running via uv.

1.  **Install LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai/).
2.  **Download Model**: Search for `qwen 3 vl 8B` in LM Studio and download it.  Wait for the download to complete.
    * (Optional) Between 4B, 8B, and 30B, choose the largest one that says "Full GPU Offload Possible".  8B is a good default.
3.  **Install uv**: As per [official instructions](https://docs.astral.sh/uv/getting-started/installation/)
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```
4.  **Run TPRS**:  
```shell
uvx --from tvas tprs
```

### Running with OpenRouter instead of LM Studio

If your machine isn't powerful enough to run models like Qwen3-VL, 
or you simply prefer faster processing, you can use a cloud provider.

1.  **Create Account**: Register on [openrouter.ai](https://openrouter.ai/) and add credits ($1 covers ~1000 photos).
2.  **Get API Key**: [Create an API key](https://openrouter.ai/settings/keys).
3.  **Save Key**: Run this command in your terminal to save your key (replace `sk-or-...` with your actual key):
    ```shell
    echo "sk-or-..." > ~/.openrouterkey
    ```
4.  **Configure Providers**: Go to [preferences](https://openrouter.ai/settings/preferences) and add `Together` and `DeepInfra` to **Ignored Providers** (to avoid reliability issues).
5.  **Install uv**: As per [official instructions](https://docs.astral.sh/uv/getting-started/installation/)
6.  **Run TPRS**:  
    ```shell
    uvx --from git+https://github.com/kagelump/vlog2 tprs --openrouter
    ``` 

#### FAQ

- **I have a ChatGPT subscription, can I use that instead?**
  No, ChatGPT subscriptions do not cover API usage.

- **Wait I can use ChatGPT for this?**
  Yeah but even GPT 5 mini is like [4x more expensive](https://openrouter.ai/openai/gpt-5-mini) than Qwen3-VL.  Regular ChatGPT is [20x more expensive](https://openrouter.ai/openai/gpt-5.2-chat).

- **Why does TPRS use LM Studio instead of MLX-VLM?**
  mlx-vlm has issues with numbers which screws up bounding boxes.  This doesn't seem to happen in LM Studio (or in most providers on OpenRouter).

## Features

### TVAS (Travel Vlog Automation System)

- **SD Card Detection**: Automatically detects camera SD cards (Sony A7C, DJI Pocket 3, iPhone, Insta360)
- **Smart Ingestion**: Copies files with SHA256 verification and organized folder structure
- **AI Analysis**: Uses Qwen3-VL (8B) via mlx-vlm for intelligent junk detection on Apple Silicon
- **OpenCV Pre-screening**: Fast blur and darkness detection before VLM analysis
- **Review UI**: Native macOS UI (Toga) for reviewing AI decisions
- **Timeline Generation**: Generates an import script for DaVinci Resolve

### TPRS (Travel Photo Rating System)

- **macOS Native App**: Runs as a standalone `.app` with a native GUI.
- **JPEG Photo Scanning**: Scans SD cards for JPEG photos.
- **AI Photo Rating**: Uses Qwen VL to analyze photo quality and rate 1-5 stars.
- **Keyword Extraction**: Automatically generates 5 descriptive keywords for each photo.
- **Caption Generation**: Creates captions to help distinguish similar high-rated photos.
- **Review Mode**: Click on processed photos to view details, including AI reasoning and subject detection.
- **Subject Detection**: Overlays a red bounding box on the detected subject in Review Mode.
- **XMP Sidecar Generation**: Outputs XMP files compatible with DxO PhotoLab and other tools, including debug data.

## Installation

### Prerequisites

- Python 3.11+
- FFmpeg (for proxy generation and video analysis)
- OpenCV (for frame extraction and pre-screening)
- Apple Silicon Mac (M1/M2/M3/M4) - **required** for mlx-vlm VLM analysis

### macOS (Homebrew)

```bash
# Install system dependencies
brew install python@3.11 ffmpeg

# Install TVAS with all features
pip install -e ".[full]"
```

The VLM model (`mlx-community/Qwen3-VL-8B-Instruct-8bit`) will be automatically downloaded from HuggingFace on first use (~6GB).

### Building the Release (Recommended for TPRS)

To build the standalone `tprs.app`:

```bash
./build_release.sh
```

This will create `tvas_release/tprs.app` which you can move to your Applications folder or run directly.

### Install from Source

```bash
git clone https://github.com/kagelump/vlog2.git
cd vlog2
pip install -e ".[full]"
```

## Usage

### TVAS - Video Analysis

#### Watch for SD Cards

```bash
tvas --watch
```

#### Process a Specific Volume

```bash
tvas --volume /Volumes/DJI_POCKET3 --project "Tokyo Day 1"
```

#### Skip UI (Auto-approve AI Decisions)

```bash
tvas --volume /Volumes/SONY_A7C --auto-approve
```

#### Disable VLM (Use OpenCV Only)

```bash
tvas --volume /Volumes/DJI_POCKET3 --no-vlm
```

### TPRS - Photo Rating System

TPRS is best used as the standalone macOS app generated by the build script.

#### Launch GUI App

Open `tprs.app` (built via `./build_release.sh`).

The GUI provides:
- **Folder Selection**: Choose a folder to scan.
- **Live Progress**: View photos as they are analyzed.
- **Recent Strip**: Side-scrolling list of recently processed photos.
- **Review Mode**: Click "View" on any recent photo to see:
    - Full resolution preview
    - Red bounding box around the primary subject
    - Detailed metadata (Rating, Keywords, Description, Raw AI Response)
- **Resume Live View**: Return to the live processing view.

#### Run via CLI (Alternative)

You can still run TPRS via the command line if installed via pip:

```bash
tprs                          # Launch GUI with folder selection dialog
tprs /Volumes/SD_CARD         # Launch GUI with pre-selected folder
```

#### Run in Headless Mode

For automated workflows or when GUI is not needed:

```bash
tprs /Volumes/SD_CARD --headless
```

This will:
- Scan for all JPEG photos on the SD card
- Analyze each photo for quality, sharpness, and composition
- Generate XMP sidecar files with:
  - `xmp:Rating` - Star rating (1-5)
  - `dc:subject` - 5 keywords describing the image
  - `dc:description` - Caption for the photo

#### Output XMP Files to Different Directory

```bash
tprs /Volumes/SD_CARD --headless --output /path/to/xmp/files
```

#### Preview Photos Without Processing

```bash
tprs /Volumes/SD_CARD --headless --dry-run
```

#### Use with DxO PhotoLab

After running `tprs`, the XMP sidecar files will be created next to your photos. When you import the photos into DxO PhotoLab:

1. The star ratings appear in the rating field
2. Keywords appear in the "Keywords" palette
3. Descriptions appear in the metadata
4. You can search for keywords like "Sunset", "Cat", or "Blurry" to find photos without looking at them

## Output Folder Structure

```
~/Movies/Vlog/
  └── 2025-11-30_Tokyo/
      ├── SonyA7C/
      ├── DJIPocket3/
      ├── iPhone11Pro/
      └── .cache/  (AI proxies, analysis JSON)
```

## Pipeline Stages

1. **Ingestion**: Copy files from SD card with verification
2. **Proxy Generation**: Create ProRes edit proxies using FFmpeg
3. **AI Analysis**: Generate clip names and suggest trim points using Qwen3 VL (8B)
4. **Timeline Generation**: Run the generated script in DaVinci Resolve to import clips and build timeline

## Configuration

| Option | Description | Default |
|--------|-------------|---------|  
| `--archival-path` | Path for archival storage (auto-detects ACASIS) | Auto-detect |
| `--proxy-path` | Path for edit proxies and cache | `~/Movies/Vlog` |
| `--model` | mlx-vlm model for VLM | `mlx-community/Qwen3-VL-8B-Instruct-8bit` |

## Development

### Source Code Structure

The project is organized into three main modules:

- **`src/tvas/`**: Travel Vlog Automation System (video processing, ingestion, timeline generation)
- **`src/tprs/`**: Travel Photo Rating System (photo analysis, rating, metadata generation)
- **`src/shared/`**: Shared utilities and prompts used by both systems

### Run Tests

```bash
pytest
```

### Run with Verbose Logging

```bash
tvas --volume /path/to/volume --verbose
```

## License

MIT

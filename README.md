# TVAS - Travel Vlog Automation System

Automate vlog ingestion, junk detection, and DaVinci Resolve import.

## Features

### TVAS (Travel Vlog Automation System)

- **SD Card Detection**: Automatically detects camera SD cards (Sony A7C, DJI Pocket 3, iPhone, Insta360)
- **Smart Ingestion**: Copies files with SHA256 verification and organized folder structure
- **AI Analysis**: Uses Qwen3-VL (8B) via mlx-vlm for intelligent junk detection on Apple Silicon
- **OpenCV Heuristics**: Fast blur and darkness detection for pre-screening
- **Review UI**: Native macOS UI (Toga) for reviewing AI decisions
- **Timeline Generation**: Creates OpenTimelineIO files for DaVinci Resolve import

### TPRS (Travel Photo Rating System)

- **JPEG Photo Scanning**: Scans SD cards for JPEG photos
- **AI Photo Rating**: Uses Qwen VL to analyze photo quality and rate 1-5 stars
- **Keyword Extraction**: Automatically generates 5 descriptive keywords for each photo
- **Caption Generation**: Creates captions to help distinguish similar high-rated photos
- **XMP Sidecar Generation**: Outputs XMP files compatible with DxO PhotoLab and other tools

## Installation

### Prerequisites

- Python 3.11+
- FFmpeg (for proxy generation and video analysis)
- Apple Silicon Mac (M1/M2/M3/M4) for VLM-based analysis with mlx-vlm

### macOS (Homebrew)

```bash
# Install system dependencies
brew install python@3.11 ffmpeg

# Install TVAS (with all features including VLM)
pip install -e ".[full]"
```

The VLM model (`mlx-community/Qwen3-VL-8B-Instruct-8bit`) will be automatically downloaded from HuggingFace on first use.

### Install from Source

```bash
git clone https://github.com/kagelump/vlog2.git
cd vlog2
pip install -e ".[full]"
```

### Optional Dependencies

- **VLM only**: `pip install -e ".[vlm]"` - For AI-powered junk detection
- **CV only**: `pip install -e ".[cv]"` - For OpenCV-based analysis
- **UI only**: `pip install -e ".[ui]"` - For the native macOS review UI

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

#### Process Photos from SD Card

```bash
tprs /Volumes/SD_CARD
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
tprs /Volumes/SD_CARD --output /path/to/xmp/files
```

#### Preview Photos Without Processing

```bash
tprs /Volumes/SD_CARD --dry-run
```

#### Use with DxO PhotoLab

After running `tprs`, the XMP sidecar files will be created next to your photos. When you import the photos into DxO PhotoLab:

1. The star ratings appear in the rating field
2. Keywords appear in the "Keywords" palette
3. Descriptions appear in the metadata
4. You can search for keywords like "Sunset", "Cat", or "Blurry" to find photos without looking at them

## Project Structure

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
2. **Proxy Generation**: Create low-res AI proxies using FFmpeg
3. **AI Analysis**: Detect junk clips using Qwen3 VL (8B) + OpenCV
4. **User Review**: Review and override AI decisions in Toga UI
5. **Timeline Generation**: Export OpenTimelineIO for DaVinci Resolve

## Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--base-path` | Base path for vlog storage | `~/Movies/Vlog` |
| `--model` | mlx-vlm model for VLM | `mlx-community/Qwen3-VL-8B-Instruct-8bit` |
| `--auto-approve` | Skip UI, approve all AI decisions | `False` |
| `--no-vlm` | Disable VLM, use OpenCV only | `False` |

## Development

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

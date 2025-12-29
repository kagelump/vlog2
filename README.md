# TVAS - Travel Vlog Automation System

Automate vlog ingestion, junk detection, and DaVinci Resolve import.

## Features

### TVAS (Travel Vlog Automation System)

- **SD Card Detection**: Automatically detects camera SD cards (Sony A7C, DJI Pocket 3, iPhone, Insta360)
- **Smart Ingestion**: Copies files with SHA256 verification and organized folder structure
- **AI Analysis**: Uses Qwen3-VL (8B) via mlx-vlm for intelligent junk detection on Apple Silicon
- **OpenCV Pre-screening**: Fast blur and darkness detection before VLM analysis
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

TPRS runs in GUI mode by default for easy folder selection and progress monitoring.

#### Launch GUI Mode (Default)

```bash
tprs                          # Launch GUI with folder selection dialog
tprs /Volumes/SD_CARD         # Launch GUI with pre-selected folder
```

The GUI will:
- Show a folder selection dialog (if no folder specified)
- Display progress with photo previews
- Show real-time analysis results
- Display recently processed photos with ratings

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
2. **Proxy Generation**: Create ProRes edit proxies using FFmpeg
3. **AI Analysis**: Generate clip names and suggest trim points using Qwen3 VL (8B)
4. **Timeline Generation**: Export OpenTimelineIO for DaVinci Resolve (review/editing done in Resolve)

## Configuration

| Option | Description | Default |
|--------|-------------|---------|  
| `--archival-path` | Path for archival storage (auto-detects ACASIS) | Auto-detect |
| `--proxy-path` | Path for edit proxies and cache | `~/Movies/Vlog` |
| `--model` | mlx-vlm model for VLM | `mlx-community/Qwen3-VL-8B-Instruct-8bit` |

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

# Project Reorganization Summary

## Overview
The project has been reorganized into three separate modules: `tvas`, `tprs`, and `shared`.

## New Directory Structure

```
src/
├── tvas/              # Travel Vlog Automation System
│   ├── __init__.py
│   ├── main.py        # Main TVAS entry point
│   ├── analysis.py    # Video clip analysis
│   ├── ingestion.py   # SD card ingestion
│   ├── timeline.py    # Timeline generation
│   ├── watcher.py     # Volume watcher
│   └── review_ui.py   # User review interface
│
├── tprs/              # Travel Photo Rating System
│   ├── __init__.py
│   ├── tprs.py        # Core TPRS functionality
│   ├── cli.py         # Command-line interface (was tprs_cli.py)
│   └── ui.py          # GUI interface (was tprs_ui.py)
│
└── shared/            # Shared utilities
    ├── __init__.py    # Common utilities (load_prompt, DEFAULT_VLM_MODEL)
    ├── proxy.py       # Proxy generation (used by TVAS)
    └── prompts/       # AI prompts directory
        ├── best_in_burst.txt
        ├── burst_similarity.txt
        ├── photo_analysis.txt
        ├── subject_sharpness.txt
        └── trim_detection.txt
```

## Key Changes

### 1. Module Separation
- **TVAS**: Video-related functionality remains in `src/tvas/`
- **TPRS**: Photo-related functionality moved to `src/tprs/`
- **Shared**: Common utilities moved to `src/shared/`

### 2. Import Updates
All files have been updated to use the new import paths:
- `from shared import DEFAULT_VLM_MODEL, load_prompt`
- `from shared.proxy import generate_proxies_batch`
- `from tprs.tprs import PhotoAnalysis, find_jpeg_photos`
- `from tprs.cli import main` (for TPRS CLI)

### 3. Configuration Files Updated
- **pyproject.toml**: Updated entry points
  - `tvas = "tvas.main:main"`
  - `tprs = "tprs.cli:main"`

- **tvas.spec**: Updated PyInstaller configuration
  - Data files: `src/shared/prompts/*.txt:shared/prompts`
  - Hidden imports: `tvas`, `tvas.main`, `shared`

- **tprs.spec**: Updated PyInstaller configuration
  - Data files: `src/shared/prompts/*.txt:shared/prompts`
  - Entry point: `src/tprs/cli.py`
  - Hidden imports: `tprs`, `tprs.cli`, `shared`

- **build_release.sh**: Updated build script paths

### 4. Test Files Updated
All test files have been updated to use new import paths:
- `tests/test_proxy.py`: Uses `shared.proxy`
- `tests/test_tprs.py`: Uses `tprs.tprs` and `shared`
- `tests/test_cli_args.py`: Uses `tprs.cli`
- `tests/test_analysis.py`: Uses `shared` for `DEFAULT_VLM_MODEL`

## Benefits

1. **Clear Separation**: TVAS and TPRS are now clearly separated modules
2. **Shared Code**: Common functionality is in one place (`shared`)
3. **Better Organization**: Easier to navigate and maintain
4. **Modular**: Each component can be developed and tested independently
5. **Scalable**: Easier to add new features to specific modules

## Migration Notes

- All old files in `src/tvas/` that were moved have been removed
- Prompts directory is now in `src/shared/prompts/`
- TPRS CLI renamed from `tprs_cli.py` to `cli.py` for cleaner naming
- TPRS UI renamed from `tprs_ui.py` to `ui.py` for cleaner naming

"""Stage 1: Ingestion & Organization

This module handles SD card detection, file copying with verification,
and organization of video files by camera source and date.
"""

import hashlib
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes.
    
    Returns:
        Formatted string (e.g., "1.5 GB", "250 MB").
    """
    size = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


class CameraType(Enum):
    """Supported camera types."""

    SONY_A7C = "SonyA7C"
    DJI_POCKET3 = "DJIPocket3"
    IPHONE_11PRO = "iPhone11Pro"
    INSTA360 = "Insta360"
    INSTA360_ULTRA_GO = "Insta360UltraGo"
    UNKNOWN = "Unknown"


@dataclass
class VideoFile:
    """Represents a video file with metadata."""

    source_path: Path
    camera_type: CameraType
    filename: str
    file_size: int
    checksum: str | None = None
    destination_path: Path | None = None


@dataclass
class IngestSession:
    """Represents an ingestion session."""

    session_id: str
    source_volume: Path
    destination_base: Path
    project_name: str
    camera_type: CameraType
    files: list[VideoFile]
    created_at: datetime


def detect_camera_type(volume_path: Path) -> CameraType:
    """Detect camera type based on directory structure and file patterns.

    Args:
        volume_path: Path to the mounted volume.

    Returns:
        CameraType enum indicating the detected camera.
    """
    # Sony A7C2: .MP4, .MTS files in PRIVATE/M4ROOT
    sony_path = volume_path / "PRIVATE" / "M4ROOT"
    if sony_path.exists():
        for ext in [".MP4", ".mp4", ".MTS", ".mts"]:
            if list(sony_path.rglob(f"*{ext}")):
                logger.info(f"Detected Sony A7C camera at {volume_path}")
                return CameraType.SONY_A7C

    # Check DCIM folder for other cameras
    dcim_path = volume_path / "DCIM"
    if dcim_path.exists():
        # Insta360 Ultra Go: VID_*.mp4 and LRV_*.lrv files in DCIM/Camera01
        camera01_path = dcim_path / "Camera01"
        if camera01_path.exists():
            vid_files = list(camera01_path.glob("VID_*.mp4")) + list(camera01_path.glob("VID_*.MP4"))
            lrv_files = list(camera01_path.glob("LRV_*.lrv")) + list(camera01_path.glob("LRV_*.LRV"))
            if vid_files or lrv_files:
                logger.info(f"Detected Insta360 Ultra Go camera at {volume_path}")
                return CameraType.INSTA360_ULTRA_GO

        # Insta360: .insv, .insp files
        for ext in [".insv", ".INSV", ".insp", ".INSP"]:
            if list(dcim_path.rglob(f"*{ext}")):
                logger.info(f"Detected Insta360 camera at {volume_path}")
                return CameraType.INSTA360

        # DJI Pocket 3: .MP4 files (DJI specific folder patterns)
        # DJI typically has folders like 100MEDIA, DJI_XXXX
        for subdir in dcim_path.iterdir():
            if subdir.is_dir() and ("DJI" in subdir.name.upper() or "MEDIA" in subdir.name.upper()):
                for ext in [".MP4", ".mp4"]:
                    if list(subdir.glob(f"*{ext}")):
                        logger.info(f"Detected DJI Pocket 3 camera at {volume_path}")
                        return CameraType.DJI_POCKET3

        # iPhone 11 Pro: .MOV files
        for ext in [".MOV", ".mov"]:
            if list(dcim_path.rglob(f"*{ext}")):
                logger.info(f"Detected iPhone 11 Pro at {volume_path}")
                return CameraType.IPHONE_11PRO

        # Generic MP4 in DCIM (fallback to DJI pattern)
        for ext in [".MP4", ".mp4"]:
            if list(dcim_path.rglob(f"*{ext}")):
                logger.info(f"Detected generic camera (assuming DJI) at {volume_path}")
                return CameraType.DJI_POCKET3

    logger.warning(f"Unknown camera type at {volume_path}")
    return CameraType.UNKNOWN


def calculate_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA256 checksum of a file.

    Args:
        file_path: Path to the file.
        chunk_size: Size of chunks to read.

    Returns:
        Hex digest of the SHA256 checksum.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_video_files(volume_path: Path, camera_type: CameraType) -> list[VideoFile]:
    """Get all video files from a volume based on camera type.

    Args:
        volume_path: Path to the mounted volume.
        camera_type: The detected camera type.

    Returns:
        List of VideoFile objects.
    """
    video_files: list[VideoFile] = []
    extensions: list[str] = []
    search_path: Path = volume_path

    if camera_type == CameraType.SONY_A7C:
        search_path = volume_path / "PRIVATE" / "M4ROOT" / "CLIP"
        extensions = [".MP4", ".mp4", ".MTS", ".mts"]
    elif camera_type == CameraType.DJI_POCKET3:
        search_path = volume_path / "DCIM"
        extensions = [".MP4", ".mp4"]
    elif camera_type == CameraType.IPHONE_11PRO:
        search_path = volume_path / "DCIM"
        extensions = [".MOV", ".mov"]
    elif camera_type == CameraType.INSTA360:
        search_path = volume_path / "DCIM"
        extensions = [".insv", ".INSV", ".insp", ".INSP"]
    elif camera_type == CameraType.INSTA360_ULTRA_GO:
        search_path = volume_path / "DCIM" / "Camera01"
        extensions = [".mp4", ".MP4"]  # Only get VID_*.mp4 files, skip LRV files
    else:
        # Unknown camera - search entire volume
        extensions = [".MP4", ".mp4", ".MOV", ".mov", ".MTS", ".mts"]

    if not search_path.exists():
        logger.warning(f"Search path does not exist: {search_path}")
        return video_files

    for ext in extensions:
        for file_path in search_path.rglob(f"*{ext}"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                video_files.append(
                    VideoFile(
                        source_path=file_path,
                        camera_type=camera_type,
                        filename=file_path.name,
                        file_size=file_path.stat().st_size,
                    )
                )

    logger.info(f"Found {len(video_files)} video files in {search_path}")
    return video_files


def create_destination_structure(
    base_path: Path,
    project_name: str,
    camera_type: CameraType,
) -> Path:
    """Create the destination directory structure.

    Structure:
        ~/Movies/Vlog/
          └── 2025-11-30_Tokyo/
              ├── SonyA7C/
              ├── DJIPocket3/
              └── .cache/

    Args:
        base_path: Base path for vlog storage (e.g., ~/Movies/Vlog).
        project_name: Name of the project (e.g., "Tokyo").
        camera_type: Camera type for subdirectory.

    Returns:
        Path to the camera-specific destination directory.
    """
    camera_folder = camera_type.value

    dest_path = base_path / project_name/ camera_folder

    dest_path.mkdir(parents=True, exist_ok=True)

    # Clean up any leftover temporary files from previous interrupted copies
    cleanup_temp_files(dest_path)

    logger.info(f"Created destination structure at {dest_path}")
    return dest_path


def cleanup_temp_files(directory: Path) -> None:
    """Remove any leftover .tmp files from interrupted copies.
    
    Args:
        directory: Directory to clean up.
    """
    tmp_files = list(directory.glob(".*.tmp"))
    if tmp_files:
        logger.info(f"Cleaning up {len(tmp_files)} temporary file(s) from previous run...")
        for tmp_file in tmp_files:
            try:
                tmp_file.unlink()
                logger.debug(f"Removed temporary file: {tmp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {tmp_file.name}: {e}")


def can_use_fast_copy(source: Path, destination: Path) -> bool:
    """Check if fast copy methods can be used (same device/filesystem).
    
    Args:
        source: Source file path.
        destination: Destination directory path.
    
    Returns:
        True if both paths are on the same device.
    """
    try:
        source_dev = source.stat().st_dev
        dest_dev = destination.stat().st_dev
        return source_dev == dest_dev
    except (OSError, AttributeError):
        return False


def try_fast_copy(source: Path, destination: Path) -> bool:
    """Attempt fast copy using system utilities.
    
    Tries in order:
    1. macOS clonefile (APFS clone - instant, CoW)
    2. cp command (may use CoW on supported filesystems)
    
    Args:
        source: Source file path.
        destination: Destination file path.
    
    Returns:
        True if fast copy succeeded, False to fall back to Python copy.
    """
    # Only try fast copy on macOS for now
    if platform.system() != "Darwin":
        return False
    
    # Try clonefile first (APFS copy-on-write clone)
    try:
        # Use clonefile via ctypes (macOS 10.12+)
        import ctypes
        libc = ctypes.CDLL(None)
        clonefile = libc.clonefile
        clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clonefile.restype = ctypes.c_int
        
        result = clonefile(
            str(source).encode('utf-8'),
            str(destination).encode('utf-8'),
            0  # flags
        )
        
        if result == 0:
            logger.debug(f"Used clonefile (instant CoW clone) for {source.name}")
            return True
    except Exception as e:
        logger.debug(f"clonefile not available or failed: {e}")
    
    # Try cp command with -c flag (clone if possible, copy otherwise)
    try:
        result = subprocess.run(
            ['cp', '-c', str(source), str(destination)],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            logger.debug(f"Used cp -c for {source.name}")
            return True
    except Exception as e:
        logger.debug(f"cp command failed: {e}")
    
    return False


def copy_file_with_progress(
    source: Path,
    destination: Path,
    progress_callback: Callable[[int, int], None] | None = None,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
) -> Path:
    """Copy a file with progress reporting using atomic write.

    Copies to a temporary file first, then atomically renames to final destination.
    This prevents partial files from appearing as complete if interrupted.
    
    Attempts fast copy methods (clonefile/cp -c) first for same-device transfers.

    Args:
        source: Source file path.
        destination: Destination file path.
        progress_callback: Optional callback(bytes_copied, total_bytes).
        chunk_size: Size of chunks for copying.

    Returns:
        Path to the copied file.
    """
    total_size = source.stat().st_size
    bytes_copied = 0

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Copy to temporary file with .tmp extension
    temp_destination = destination.parent / f".{destination.name}.tmp"

    try:
        # Try fast copy first if on same device
        if can_use_fast_copy(source, destination.parent):
            logger.debug(f"Attempting fast copy for {source.name}")
            if try_fast_copy(source, temp_destination):
                # Fast copy succeeded - report 100% progress
                if progress_callback:
                    progress_callback(total_size, total_size)
                
                # Copy metadata
                shutil.copystat(source, temp_destination)
                
                # Atomic rename
                temp_destination.replace(destination)
                logger.debug(f"Fast-copied {source} to {destination}")
                return destination
        
        # Fall back to chunk-based copy with progress
        with open(source, "rb") as src, open(temp_destination, "wb") as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                dst.write(chunk)
                bytes_copied += len(chunk)
                if progress_callback:
                    progress_callback(bytes_copied, total_size)

        # Copy metadata (timestamps, etc.)
        shutil.copystat(source, temp_destination)

        # Atomic rename - if interrupted before this, temp file remains
        temp_destination.replace(destination)

        logger.debug(f"Copied {source} to {destination}")
        return destination
    except Exception as e:
        # Clean up temp file on error
        if temp_destination.exists():
            temp_destination.unlink()
            logger.debug(f"Cleaned up temporary file: {temp_destination}")
        raise


def verify_copy(source: Path, destination: Path) -> bool:
    """Verify that a file was copied correctly using SHA256.

    Args:
        source: Original source file.
        destination: Copied destination file.

    Returns:
        True if checksums match, False otherwise.
    """
    if not destination.exists():
        logger.error(f"Destination file does not exist: {destination}")
        return False

    source_checksum = calculate_sha256(source)
    dest_checksum = calculate_sha256(destination)

    if source_checksum == dest_checksum:
        logger.debug(f"Checksum verified for {destination.name}")
        return True
    else:
        logger.error(f"Checksum mismatch for {destination.name}")
        return False


def ingest_volume(
    volume_path: Path,
    destination_base: Path,
    project_name: str,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> IngestSession:
    """Ingest all video files from a volume.

    Args:
        volume_path: Path to the mounted volume.
        destination_base: Base path for vlog storage.
        project_name: Name of the project.
        progress_callback: Optional callback(filename, file_index, total_files).

    Returns:
        IngestSession with details of the ingestion.
    """
    camera_type = detect_camera_type(volume_path)
    video_files = get_video_files(volume_path, camera_type)

    if not video_files:
        logger.warning(f"No video files found on {volume_path}")

    dest_path = create_destination_structure(destination_base, project_name, camera_type)

    session = IngestSession(
        session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        source_volume=volume_path,
        destination_base=destination_base,
        project_name=project_name,
        camera_type=camera_type,
        files=video_files,
        created_at=datetime.now(),
    )

    for i, video_file in enumerate(video_files):
        if progress_callback:
            progress_callback(video_file.filename, i + 1, len(video_files))

        dest_file = dest_path / video_file.filename
        file_size_str = format_file_size(video_file.file_size)
        
        # Check if file already exists and is valid
        if dest_file.exists():
            dest_stat = dest_file.stat()
            source_stat = video_file.source_path.stat()
            
            # Quick validation: size and modification time must match
            if dest_stat.st_size == video_file.file_size and dest_stat.st_mtime >= source_stat.st_mtime:
                logger.info(f"Skipping {video_file.filename} ({file_size_str}) - already copied")
                video_file.destination_path = dest_file
                # Note: checksum will be None for skipped files to save time
                continue
            elif dest_stat.st_size == video_file.file_size:
                # Size matches but mtime doesn't - source may be newer, re-copy
                logger.warning(f"File exists but may be outdated (mtime mismatch), re-copying: {video_file.filename}")
                dest_file.unlink()
            else:
                logger.warning(f"File exists but size mismatch (expected {file_size_str}, got {format_file_size(dest_stat.st_size)}), re-copying: {video_file.filename}")
                dest_file.unlink()  # Remove incomplete file
        
        logger.info(f"Copying {video_file.filename} ({file_size_str})...")
        copy_file_with_progress(video_file.source_path, dest_file)

        # Only verify checksum for freshly copied files
        if verify_copy(video_file.source_path, dest_file):
            video_file.destination_path = dest_file
            video_file.checksum = calculate_sha256(dest_file)
        else:
            logger.error(f"Failed to verify {video_file.filename}")

    logger.info(f"Ingestion complete: {len(video_files)} files processed")
    return session

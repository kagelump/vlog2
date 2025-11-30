"""Stage 1: Ingestion & Organization

This module handles SD card detection, file copying with verification,
and organization of video files by camera source and date.
"""

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class CameraType(Enum):
    """Supported camera types."""

    SONY_A7C = "SonyA7C"
    DJI_POCKET3 = "DJIPocket3"
    IPHONE_11PRO = "iPhone11Pro"
    INSTA360 = "Insta360"
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
        search_path = volume_path / "PRIVATE" / "M4ROOT"
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
    else:
        # Unknown camera - search entire volume
        extensions = [".MP4", ".mp4", ".MOV", ".mov", ".MTS", ".mts"]

    if not search_path.exists():
        logger.warning(f"Search path does not exist: {search_path}")
        return video_files

    for ext in extensions:
        for file_path in search_path.rglob(f"*{ext}"):
            if file_path.is_file():
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
    date_str = datetime.now().strftime("%Y-%m-%d")
    project_folder = f"{date_str}_{project_name}"
    camera_folder = camera_type.value

    dest_path = base_path / project_folder / camera_folder
    cache_path = base_path / project_folder / ".cache"

    dest_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created destination structure at {dest_path}")
    return dest_path


def copy_file_with_progress(
    source: Path,
    destination: Path,
    progress_callback: Callable[[int, int], None] | None = None,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
) -> Path:
    """Copy a file with progress reporting.

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

    with open(source, "rb") as src, open(destination, "wb") as dst:
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)
            bytes_copied += len(chunk)
            if progress_callback:
                progress_callback(bytes_copied, total_size)

    # Copy metadata (timestamps, etc.)
    shutil.copystat(source, destination)

    logger.debug(f"Copied {source} to {destination}")
    return destination


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
        copy_file_with_progress(video_file.source_path, dest_file)

        if verify_copy(video_file.source_path, dest_file):
            video_file.destination_path = dest_file
            video_file.checksum = calculate_sha256(dest_file)
        else:
            logger.error(f"Failed to verify {video_file.filename}")

    logger.info(f"Ingestion complete: {len(video_files)} files processed")
    return session

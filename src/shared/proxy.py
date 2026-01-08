"""Stage 2: Proxy Generation

This module handles proxy video generation using FFmpeg with hardware acceleration.
Generates edit proxies (ProRes) for smooth editing in DaVinci Resolve.
"""

import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds.
    
    Returns:
        Formatted string (e.g., "1:23:45", "2:15").
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


@dataclass
class ProxyResult:
    """Result of proxy generation for a single file."""

    source_path: Path
    proxy_path: Path | None
    success: bool
    duration_seconds: float
    error_message: str | None = None


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system.

    Returns:
        True if FFmpeg is available, False otherwise.
    """
    return shutil.which("ffmpeg") is not None


def check_videotoolbox_available() -> bool:
    """Check if VideoToolbox hardware acceleration is available (macOS).

    Returns:
        True if VideoToolbox is available, False otherwise.
    """
    if not check_ffmpeg_available():
        return False

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "h264_videotoolbox" in result.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def build_edit_proxy_command(
    source_path: Path,
    output_path: Path,
) -> list[str]:
    """Build FFmpeg command for edit proxy generation (h264).

    Edit Proxy specs:
    - Format: h264
    - Resolution: Maintains original

    Args:
        source_path: Path to source video.
        output_path: Path for output proxy.

    Returns:
        List of command arguments for subprocess.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-hwaccel", "videotoolbox",
        "-i", str(source_path),
        "-vf", "scale=-2:1080: flags=fast_bilinear",      # Resize to 1080p height, maintain aspect ratio
        "-c:v", "h264_videotoolbox",
        "-profile:v", "high",
        "-b:v", "6000k",
        "-allow_sw", "1",
        "-c:a", "copy",
        "-map_metadata", "0",
        "-map_metadata:s:v", "0:s:v",
        "-map_metadata:s:a", "0:s:a",
        "-movflags", "use_metadata_tags",
        str(output_path),
    ]
    return cmd


def generate_proxy(
    source_path: Path,
    output_dir: Path,
) -> ProxyResult:
    """Generate an edit proxy video for a single file.

    Args:
        source_path: Path to source video.
        output_dir: Directory for output proxy.

    Returns:
        ProxyResult with details of the operation.
    """

    # Determine output filename
    output_path = output_dir / (source_path.stem + ".mp4")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if proxy already exists with correct duration
    if output_path.exists():
        source_duration = get_video_duration(source_path)
        proxy_duration = get_video_duration(output_path)
        
        if source_duration and proxy_duration:
            # Allow 2% tolerance for duration differences (framerate adjustments, encoding)
            duration_diff = abs(source_duration - proxy_duration)
            tolerance = source_duration * 0.02  # 2% tolerance
            
            if duration_diff <= tolerance:
                logger.info(f"Skipping {source_path.name} - edit proxy already exists with correct duration ({format_duration(proxy_duration)})")
                return ProxyResult(
                    source_path=source_path,
                    proxy_path=output_path,
                    success=True,
                    duration_seconds=0.0,  # No processing time since skipped
                )
            else:
                logger.warning(f"Proxy exists but duration mismatch (source: {format_duration(source_duration)}, proxy: {format_duration(proxy_duration)}), regenerating...")
                output_path.unlink()  # Remove incorrect proxy
        else:
            # Can't determine duration, regenerate to be safe
            logger.warning(f"Proxy exists but duration check failed, regenerating...")
            output_path.unlink()

    # Build edit proxy command
    cmd = build_edit_proxy_command(source_path, output_path)

    # Get video duration for logging
    video_duration = get_video_duration(source_path)
    duration_str = f" ({format_duration(video_duration)})" if video_duration else ""
    
    logger.info(f"Generating edit proxy for {source_path.name}{duration_str}")
    logger.debug(f"FFmpeg command: {' '.join(cmd)}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )

        duration = time.time() - start_time

        if result.returncode == 0 and output_path.exists():
            # Copy file timestamps (mtime, atime) from source to proxy
            try:
                shutil.copystat(source_path, output_path)
            except Exception as e:
                logger.warning(f"Failed to copy timestamps for {output_path.name}: {e}")

            logger.info(f"Successfully generated proxy in {duration:.1f}s: {output_path.name}")
            return ProxyResult(
                source_path=source_path,
                proxy_path=output_path,
                success=True,
                duration_seconds=duration,
            )
        else:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            logger.error(f"FFmpeg failed for {source_path.name}: {error_msg}")
            return ProxyResult(
                source_path=source_path,
                proxy_path=None,
                success=False,
                duration_seconds=duration,
                error_message=error_msg,
            )

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"FFmpeg timed out for {source_path.name}")
        return ProxyResult(
            source_path=source_path,
            proxy_path=None,
            success=False,
            duration_seconds=duration,
            error_message="Process timed out after 600 seconds",
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error generating proxy for {source_path.name}: {e}")
        return ProxyResult(
            source_path=source_path,
            proxy_path=None,
            success=False,
            duration_seconds=duration,
            error_message=str(e),
        )


def generate_proxies_batch(
    source_files: list[Path],
    output_dir: Path,
    max_workers: int = 4,
    progress_callback: Any = None,
) -> list[ProxyResult]:
    """Generate edit proxy videos for a batch of files with parallel processing.

    Args:
        source_files: List of source video paths.
        output_dir: Directory for output proxies.
        max_workers: Maximum number of concurrent encoding jobs (default: 3).
        progress_callback: Optional callback(current, total, result).

    Returns:
        List of ProxyResult for each file.
    """
    results: list[ProxyResult] = []
    total_files = len(source_files)
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_source = {
            executor.submit(generate_proxy, source_path, output_dir): source_path
            for source_path in source_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_source):
            source_path = future_to_source[future]
            completed_count += 1
            try:
                result = future.result()
                results.append(result)
                if progress_callback:
                    progress_callback(completed_count, total_files, result)
            except Exception as e:
                logger.error(f"Unexpected error processing {source_path.name}: {e}")
                error_result = ProxyResult(
                    source_path=source_path,
                    proxy_path=None,
                    success=False,
                    duration_seconds=0.0,
                    error_message=str(e),
                )
                results.append(error_result)
                if progress_callback:
                    progress_callback(completed_count, total_files, error_result)

    successful = sum(1 for r in results if r.success)
    logger.info(f"Proxy generation complete: {successful}/{len(results)} successful")

    return results


def get_video_duration(video_path: Path) -> float | None:
    """Get the duration of a video file in seconds.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds, or None if unable to determine.
    """
    if not check_ffmpeg_available():
        return None

    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        pass

    return None


def get_video_info(video_path: Path) -> dict[str, Any] | None:
    """Get metadata information about a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with video metadata, or None if unable to read.
    """
    if not check_ffmpeg_available():
        return None

    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        pass

    return None


def extract_frame(
    video_path: Path,
    timestamp: float,
    output_path: Path,
    max_dimension: int | None = None
) -> bool:
    """Extract a single frame from video at timestamp using FFmpeg.

    Args:
        video_path: Path to the video file.
        timestamp: Timestamp in seconds.
        output_path: Path to save the extracted frame (JPEG).
        max_dimension: Optional maximum dimension (width/height) to resize to.

    Returns:
        True if successful, False otherwise.
    """
    if not check_ffmpeg_available():
        return False

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
    ]

    if max_dimension:
        # Scale to max_dimension while maintaining aspect ratio
        cmd.extend(["-vf", f"scale='min({max_dimension},iw)':-1"])

    cmd.append(str(output_path))

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Failed to extract frame from {video_path} at {timestamp}: {e}")
        return False

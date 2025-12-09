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
from dataclasses import dataclass
from enum import Enum
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


class ProxyType(Enum):
    """Types of proxy videos."""

    EDIT_PROXY = "edit_proxy"  # ProRes Proxy - for smooth editing


@dataclass
class ProxyConfig:
    """Configuration for proxy generation."""

    use_hardware_accel: bool = True  # Use videotoolbox on macOS
    batch_size: int = 5  # Files per batch
    cooldown_seconds: int = 30  # Cooldown between batches
    max_temp_celsius: int = 95  # Maximum CPU temperature


@dataclass
class ProxyResult:
    """Result of proxy generation for a single file."""

    source_path: Path
    proxy_path: Path | None
    proxy_type: ProxyType
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
    """Build FFmpeg command for edit proxy generation (ProRes Proxy).

    Edit Proxy specs:
    - Format: ProRes Proxy (422 Proxy)
    - Resolution: Maintains original
    - For smooth scrubbing in DaVinci Resolve

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
        "-i",
        str(source_path),
        "-c:v",
        "prores_ks",
        "-profile:v",
        "0",  # ProRes 422 Proxy
        "-c:a",
        "pcm_s16le",  # Preserve audio
        str(output_path),
    ]
    return cmd


def generate_proxy(
    source_path: Path,
    output_dir: Path,
    config: ProxyConfig | None = None,
) -> ProxyResult:
    """Generate an edit proxy video for a single file.

    Args:
        source_path: Path to source video.
        output_dir: Directory for output proxy.
        config: Proxy configuration (uses defaults if None).

    Returns:
        ProxyResult with details of the operation.
    """
    if config is None:
        config = ProxyConfig()

    # Determine output filename
    output_path = output_dir / (source_path.stem + "_edit_proxy.mov")

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
                    proxy_type=ProxyType.EDIT_PROXY,
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
            timeout=600,  # 10 minute timeout
        )

        duration = time.time() - start_time

        if result.returncode == 0 and output_path.exists():
            logger.info(f"Successfully generated proxy in {duration:.1f}s: {output_path.name}")
            return ProxyResult(
                source_path=source_path,
                proxy_path=output_path,
                proxy_type=ProxyType.EDIT_PROXY,
                success=True,
                duration_seconds=duration,
            )
        else:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            logger.error(f"FFmpeg failed for {source_path.name}: {error_msg}")
            return ProxyResult(
                source_path=source_path,
                proxy_path=None,
                proxy_type=ProxyType.EDIT_PROXY,
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
            proxy_type=ProxyType.EDIT_PROXY,
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
            proxy_type=ProxyType.EDIT_PROXY,
            success=False,
            duration_seconds=duration,
            error_message=str(e),
        )


def generate_proxies_batch(
    source_files: list[Path],
    output_dir: Path,
    config: ProxyConfig | None = None,
) -> list[ProxyResult]:
    """Generate edit proxy videos for a batch of files with thermal management.

    Processes files in batches with cooldown periods to manage thermal load.

    Args:
        source_files: List of source video paths.
        output_dir: Directory for output proxies.
        config: Proxy configuration.

    Returns:
        List of ProxyResult for each file.
    """
    if config is None:
        config = ProxyConfig()

    results: list[ProxyResult] = []
    batch_count = 0
    processed_count = 0  # Track actually processed (not skipped) videos

    for i, source_path in enumerate(source_files):
        # Generate proxy
        result = generate_proxy(source_path, output_dir, config)
        results.append(result)

        batch_count += 1
        
        # Only count as processed if ffmpeg actually ran (duration > 0)
        if result.success and result.duration_seconds > 0:
            processed_count += 1

        # Thermal management: cooldown after each batch, scaled by actual work done
        # No cooldown needed if using hardware acceleration (GPU doesn't overheat like CPU)
        if batch_count >= config.batch_size and i < len(source_files) - 1:
            # Skip cooldown if hardware acceleration is enabled
            if config.use_hardware_accel and check_videotoolbox_available():
                logger.info(f"Batch complete ({processed_count} processed, {batch_count - processed_count} skipped). No cooldown needed (hardware acceleration).")
            # Scale cooldown: 0 processed = no cooldown, full batch = full cooldown
            elif processed_count > 0:
                cooldown_ratio = processed_count / config.batch_size
                scaled_cooldown = int(config.cooldown_seconds * cooldown_ratio)
                logger.info(f"Batch complete ({processed_count} processed, {batch_count - processed_count} skipped). Cooling down for {scaled_cooldown}s...")
                time.sleep(scaled_cooldown)
            else:
                logger.info(f"Batch complete (all {batch_count} skipped). No cooldown needed.")
            
            batch_count = 0
            processed_count = 0

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

"""Stage 2: Proxy Generation

This module handles proxy video generation using FFmpeg with hardware acceleration.
Generates AI proxies (low-res for VLM inference) and optional edit proxies (ProRes).
"""

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


class ProxyType(Enum):
    """Types of proxy videos."""

    AI_PROXY = "ai_proxy"  # 640px wide, 10fps, 500kbps - for VLM analysis
    EDIT_PROXY = "edit_proxy"  # ProRes Proxy - for smooth editing


@dataclass
class ProxyConfig:
    """Configuration for proxy generation."""

    width: int = 640  # AI proxy width
    framerate: int = 10  # AI proxy framerate
    bitrate: str = "500k"  # AI proxy bitrate
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


def build_ai_proxy_command(
    source_path: Path,
    output_path: Path,
    config: ProxyConfig,
) -> list[str]:
    """Build FFmpeg command for AI proxy generation.

    AI Proxy specs:
    - Resolution: 640px wide (maintains aspect ratio)
    - Framerate: 10fps
    - Bitrate: 500kbps
    - Audio: Stripped

    Args:
        source_path: Path to source video.
        output_path: Path for output proxy.
        config: Proxy configuration.

    Returns:
        List of command arguments for subprocess.
    """
    cmd = ["ffmpeg", "-hide_banner", "-y"]

    # Input
    cmd.extend(["-i", str(source_path)])

    # Video filters: scale to width, reduce framerate
    vf = f"scale={config.width}:-2,fps={config.framerate}"
    cmd.extend(["-vf", vf])

    # Encoder selection
    if config.use_hardware_accel and check_videotoolbox_available():
        # macOS hardware acceleration
        cmd.extend(["-c:v", "h264_videotoolbox"])
        cmd.extend(["-b:v", config.bitrate])
    else:
        # Software encoding fallback
        cmd.extend(["-c:v", "libx264"])
        cmd.extend(["-preset", "fast"])
        cmd.extend(["-crf", "28"])
        cmd.extend(["-b:v", config.bitrate])

    # No audio for AI analysis
    cmd.extend(["-an"])

    # Output format
    cmd.extend(["-movflags", "+faststart"])
    cmd.append(str(output_path))

    return cmd


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
    proxy_type: ProxyType = ProxyType.AI_PROXY,
    config: ProxyConfig | None = None,
) -> ProxyResult:
    """Generate a proxy video for a single file.

    Args:
        source_path: Path to source video.
        output_dir: Directory for output proxy.
        proxy_type: Type of proxy to generate.
        config: Proxy configuration (uses defaults if None).

    Returns:
        ProxyResult with details of the operation.
    """
    if config is None:
        config = ProxyConfig()

    # Determine output filename
    suffix = "_ai_proxy.mp4" if proxy_type == ProxyType.AI_PROXY else "_edit_proxy.mov"
    output_path = output_dir / (source_path.stem + suffix)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build appropriate command
    if proxy_type == ProxyType.AI_PROXY:
        cmd = build_ai_proxy_command(source_path, output_path, config)
    else:
        cmd = build_edit_proxy_command(source_path, output_path)

    logger.info(f"Generating {proxy_type.value} for {source_path.name}")
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
                proxy_type=proxy_type,
                success=True,
                duration_seconds=duration,
            )
        else:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            logger.error(f"FFmpeg failed for {source_path.name}: {error_msg}")
            return ProxyResult(
                source_path=source_path,
                proxy_path=None,
                proxy_type=proxy_type,
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
            proxy_type=proxy_type,
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
            proxy_type=proxy_type,
            success=False,
            duration_seconds=duration,
            error_message=str(e),
        )


def generate_proxies_batch(
    source_files: list[Path],
    output_dir: Path,
    proxy_type: ProxyType = ProxyType.AI_PROXY,
    config: ProxyConfig | None = None,
) -> list[ProxyResult]:
    """Generate proxy videos for a batch of files with thermal management.

    Processes files in batches with cooldown periods to manage thermal load.

    Args:
        source_files: List of source video paths.
        output_dir: Directory for output proxies.
        proxy_type: Type of proxy to generate.
        config: Proxy configuration.

    Returns:
        List of ProxyResult for each file.
    """
    if config is None:
        config = ProxyConfig()

    results: list[ProxyResult] = []
    batch_count = 0

    for i, source_path in enumerate(source_files):
        # Generate proxy
        result = generate_proxy(source_path, output_dir, proxy_type, config)
        results.append(result)

        batch_count += 1

        # Thermal management: cooldown after each batch
        if batch_count >= config.batch_size and i < len(source_files) - 1:
            logger.info(f"Batch complete. Cooling down for {config.cooldown_seconds}s...")
            time.sleep(config.cooldown_seconds)
            batch_count = 0

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
            import json

            return json.loads(result.stdout)
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        pass

    return None

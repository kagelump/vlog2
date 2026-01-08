"""Shared FFmpeg utilities for video processing."""

import logging
import platform
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Cached codec detection result
_cached_video_codec: Optional[list[str]] = None

def detect_best_video_codec() -> list[str]:
    """Detect the best available video codec with hardware acceleration.
    
    Returns:
        List of ffmpeg arguments for video encoding (e.g., ["-c:v", "h264_videotoolbox", "-b:v", "2000k"])
    """
    global _cached_video_codec
    
    if _cached_video_codec:
        return _cached_video_codec

    system = platform.system()
    candidates = []

    if system == "Darwin":
        candidates.append(["h264_videotoolbox", "-b:v", "2000k", "-allow_sw", "1"])
    elif system == "Linux" or system == "Windows":
        candidates.append(["h264_nvenc", "-preset", "p1", "-b:v", "2000k"])
        candidates.append(["h264_qsv", "-global_quality", "25", "-load_plugin", "hevc_hw"])
        candidates.append(["h264_vaapi"])
    
    # Software fallbacks
    candidates.append(["libx264", "-preset", "ultrafast", "-crf", "28"])
    candidates.append(["libopenh264", "-b:v", "2000k"])

    # Test codecs
    for codec_args in candidates:
        codec_name = codec_args[0]
        try:
            # Minimal test to check if encoder exists and works
            cmd = [
                "ffmpeg", "-v", "error", 
                "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1", 
                "-c:v", codec_name, 
                "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Detected usable video codec: {codec_name}")
                _cached_video_codec = ["-c:v"] + codec_args
                return _cached_video_codec
        except Exception:
            continue

    # Ultimate fallback - no codec specified, let ffmpeg choose
    logger.warning("No optimized video codec found, falling back to default.")
    _cached_video_codec = []
    return _cached_video_codec


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_thumbnail(video_path: Path, output_path: Path, timestamp: float = 1.0) -> bool:
    """Extract a thumbnail from a video file."""
    try:
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y", "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1", "-q:v", "2",
            "-vf", "scale=320:-1",
            str(output_path)
        ]
        
        # Don't check=True to handle potential ffmpeg warnings/errors gracefully
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        
        if result.returncode != 0:
            logger.warning(f"ffmpeg returned {result.returncode} for {video_path}")
            
        return output_path.exists()
    except Exception as e:
        logger.error(f"Failed to extract thumbnail: {e}")
        return False

"""Stage 3: AI Analysis (Junk Detection)

This module handles AI-powered junk detection using Qwen 2.5 VL via Ollama,
combined with OpenCV heuristics for blur and darkness detection.
"""

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# OpenCV is optional - will gracefully degrade if not available
try:
    import cv2
    import numpy as np

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None
    np = None


class JunkReason(Enum):
    """Reasons a clip may be flagged as junk."""

    BLUR = "blur"
    DARKNESS = "darkness"
    LENS_CAP = "lens_cap"
    GROUND = "pointing_at_ground"
    ACCIDENTAL = "accidental_trigger"
    LOW_AUDIO = "low_audio"
    MULTIPLE = "multiple_issues"


class ConfidenceLevel(Enum):
    """Confidence level of junk detection."""

    HIGH = "high"  # OpenCV + VLM agree
    MEDIUM = "medium"  # VLM only
    LOW = "low"  # OpenCV only


class ClipDecision(Enum):
    """Decision for a clip."""

    KEEP = "keep"
    REJECT = "reject"
    REVIEW = "review"  # Uncertain, needs user review


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""

    frame_index: int
    timestamp_seconds: float
    is_junk: bool
    junk_reasons: list[JunkReason] = field(default_factory=list)
    blur_score: float | None = None
    brightness_score: float | None = None
    vlm_response: str | None = None


@dataclass
class ClipAnalysis:
    """Complete analysis result for a video clip."""

    source_path: Path
    proxy_path: Path | None
    duration_seconds: float
    frame_analyses: list[FrameAnalysis]
    decision: ClipDecision
    confidence: ConfidenceLevel
    junk_reasons: list[JunkReason]
    suggested_in_point: float | None = None
    suggested_out_point: float | None = None
    mean_audio_db: float | None = None
    vlm_summary: str | None = None


def check_ollama_available() -> bool:
    """Check if Ollama is available on the system.

    Returns:
        True if Ollama is available and running.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def check_model_available(model_name: str = "qwen2.5-vl:7b") -> bool:
    """Check if the specified model is available in Ollama.

    Args:
        model_name: Name of the Ollama model.

    Returns:
        True if the model is available.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return model_name in result.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def extract_frames(
    video_path: Path,
    output_dir: Path,
    timestamps: list[float] | None = None,
) -> list[tuple[float, Path]]:
    """Extract frames from a video at specified timestamps.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        timestamps: List of timestamps in seconds. If None, uses default sampling.

    Returns:
        List of (timestamp, frame_path) tuples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[float, Path]] = []

    if not OPENCV_AVAILABLE:
        logger.warning("OpenCV not available - using FFmpeg for frame extraction")
        return _extract_frames_ffmpeg(video_path, output_dir, timestamps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return frames

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Default timestamps: start (0-3s) and end (last 3s)
    if timestamps is None:
        timestamps = [0, 1, 2, 3]
        if duration > 6:
            timestamps.extend([duration - 3, duration - 2, duration - 1, max(0, duration - 0.1)])

    for ts in timestamps:
        frame_num = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            frame_path = output_dir / f"frame_{ts:.1f}s.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames.append((ts, frame_path))
            logger.debug(f"Extracted frame at {ts}s")

    cap.release()
    return frames


def _extract_frames_ffmpeg(
    video_path: Path,
    output_dir: Path,
    timestamps: list[float] | None = None,
) -> list[tuple[float, Path]]:
    """Extract frames using FFmpeg (fallback when OpenCV unavailable).

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        timestamps: List of timestamps in seconds.

    Returns:
        List of (timestamp, frame_path) tuples.
    """
    frames: list[tuple[float, Path]] = []

    if timestamps is None:
        timestamps = [0, 1, 2, 3]

    for ts in timestamps:
        frame_path = output_dir / f"frame_{ts:.1f}s.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-ss",
            str(ts),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and frame_path.exists():
                frames.append((ts, frame_path))
        except subprocess.SubprocessError:
            logger.warning(f"Failed to extract frame at {ts}s")

    return frames


def calculate_blur_score(image_path: Path) -> float | None:
    """Calculate blur score using Laplacian variance.

    Higher score = sharper image, lower score = more blur.

    Args:
        image_path: Path to the image file.

    Returns:
        Blur score (Laplacian variance), or None if unable to calculate.
    """
    if not OPENCV_AVAILABLE:
        return None

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    variance = laplacian.var()

    return float(variance)


def calculate_brightness_score(image_path: Path) -> float | None:
    """Calculate brightness score using HSV value channel.

    Returns mean brightness (0-255).

    Args:
        image_path: Path to the image file.

    Returns:
        Mean brightness score, or None if unable to calculate.
    """
    if not OPENCV_AVAILABLE:
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()

    return float(brightness)


def analyze_frame_opencv(frame_path: Path) -> dict:
    """Analyze a frame using OpenCV heuristics.

    Args:
        frame_path: Path to the frame image.

    Returns:
        Dictionary with analysis results.
    """
    blur_score = calculate_blur_score(frame_path)
    brightness_score = calculate_brightness_score(frame_path)

    is_blurry = blur_score is not None and blur_score < 100
    is_dark = brightness_score is not None and brightness_score < 30

    reasons = []
    if is_blurry:
        reasons.append(JunkReason.BLUR)
    if is_dark:
        reasons.append(JunkReason.DARKNESS)

    return {
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "is_blurry": is_blurry,
        "is_dark": is_dark,
        "junk_reasons": reasons,
    }


def analyze_frame_vlm(
    frame_path: Path,
    model_name: str = "qwen2.5-vl:7b",
) -> dict:
    """Analyze a frame using Vision Language Model via Ollama.

    Args:
        frame_path: Path to the frame image.
        model_name: Name of the Ollama model to use.

    Returns:
        Dictionary with VLM analysis results.
    """
    prompt = """Analyze this video frame for quality issues. 
Look for: blur/motion blur, camera pointing at ground, lens cap covering lens, 
extremely dark/black frames, accidental recording triggers.

Respond with ONLY valid JSON in this exact format:
{"is_junk": true/false, "reason": "brief explanation", "issues": ["list", "of", "issues"]}

Be conservative - only mark as junk if there are clear quality issues."""

    try:
        # Use Ollama CLI with image
        cmd = [
            "ollama",
            "run",
            model_name,
            prompt,
        ]

        # For models that support images, we'd pass the image differently
        # This is a simplified version - in production would use Ollama's API
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            input=str(frame_path),
        )

        if result.returncode == 0:
            response = result.stdout.strip()

            # Try to parse JSON from response
            try:
                # Find JSON in response (model might add extra text)
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    data = json.loads(json_str)
                    return {
                        "is_junk": data.get("is_junk", False),
                        "reason": data.get("reason", ""),
                        "issues": data.get("issues", []),
                        "raw_response": response,
                    }
            except json.JSONDecodeError:
                pass

            # Fallback: simple keyword detection
            is_junk = any(word in response.lower() for word in ["junk", "blur", "dark", "ground", "accidental"])
            return {
                "is_junk": is_junk,
                "reason": response[:200],
                "issues": [],
                "raw_response": response,
            }

    except subprocess.TimeoutExpired:
        logger.warning(f"VLM analysis timed out for {frame_path}")
    except subprocess.SubprocessError as e:
        logger.warning(f"VLM analysis failed for {frame_path}: {e}")

    return {
        "is_junk": False,
        "reason": "VLM analysis unavailable",
        "issues": [],
        "raw_response": None,
    }


def get_audio_level(video_path: Path) -> float | None:
    """Get mean audio level of a video using FFmpeg volumedetect.

    Args:
        video_path: Path to the video file.

    Returns:
        Mean volume in dB, or None if unable to determine.
    """
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(video_path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Parse mean_volume from stderr
        for line in result.stderr.split("\n"):
            if "mean_volume:" in line:
                # Extract number: "mean_volume: -30.5 dB"
                parts = line.split("mean_volume:")[-1].strip().split()
                if parts:
                    return float(parts[0])

    except (subprocess.SubprocessError, ValueError, IndexError):
        pass

    return None


def analyze_clip(
    source_path: Path,
    proxy_path: Path | None = None,
    use_vlm: bool = True,
    model_name: str = "qwen2.5-vl:7b",
) -> ClipAnalysis:
    """Perform complete analysis of a video clip.

    Combines OpenCV heuristics and VLM analysis to detect junk clips.

    Args:
        source_path: Path to the original video file.
        proxy_path: Path to the AI proxy (preferred for analysis).
        use_vlm: Whether to use VLM for analysis.
        model_name: Ollama model name for VLM analysis.

    Returns:
        ClipAnalysis with complete analysis results.
    """
    video_to_analyze = proxy_path if proxy_path and proxy_path.exists() else source_path

    # Get video duration
    from tvas.proxy import get_video_duration

    duration = get_video_duration(video_to_analyze) or 0

    # Extract frames for analysis
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        frames = extract_frames(video_to_analyze, temp_path)

        frame_analyses: list[FrameAnalysis] = []
        all_junk_reasons: list[JunkReason] = []

        for timestamp, frame_path in frames:
            # OpenCV analysis
            cv_result = analyze_frame_opencv(frame_path)

            # VLM analysis (if enabled and available)
            vlm_result = None
            if use_vlm and check_ollama_available():
                vlm_result = analyze_frame_vlm(frame_path, model_name)

            # Combine results
            is_junk = False
            junk_reasons: list[JunkReason] = []

            # OpenCV findings
            if cv_result.get("is_blurry"):
                junk_reasons.append(JunkReason.BLUR)
                is_junk = True
            if cv_result.get("is_dark"):
                junk_reasons.append(JunkReason.DARKNESS)
                is_junk = True

            # VLM findings
            if vlm_result and vlm_result.get("is_junk"):
                is_junk = True
                for issue in vlm_result.get("issues", []):
                    issue_lower = issue.lower()
                    if "blur" in issue_lower:
                        if JunkReason.BLUR not in junk_reasons:
                            junk_reasons.append(JunkReason.BLUR)
                    elif "dark" in issue_lower:
                        if JunkReason.DARKNESS not in junk_reasons:
                            junk_reasons.append(JunkReason.DARKNESS)
                    elif "ground" in issue_lower:
                        junk_reasons.append(JunkReason.GROUND)
                    elif "lens" in issue_lower or "cap" in issue_lower:
                        junk_reasons.append(JunkReason.LENS_CAP)
                    elif "accidental" in issue_lower:
                        junk_reasons.append(JunkReason.ACCIDENTAL)

            frame_analyses.append(
                FrameAnalysis(
                    frame_index=len(frame_analyses),
                    timestamp_seconds=timestamp,
                    is_junk=is_junk,
                    junk_reasons=junk_reasons,
                    blur_score=cv_result.get("blur_score"),
                    brightness_score=cv_result.get("brightness_score"),
                    vlm_response=vlm_result.get("raw_response") if vlm_result else None,
                )
            )

            all_junk_reasons.extend(junk_reasons)

    # Get audio level
    mean_audio_db = get_audio_level(source_path)

    # Check for low audio (possible accidental recording)
    if mean_audio_db is not None and mean_audio_db < -40:
        all_junk_reasons.append(JunkReason.LOW_AUDIO)

    # Decision logic
    decision, confidence, in_point, out_point = _make_decision(
        frame_analyses, duration, mean_audio_db
    )

    # Deduplicate reasons
    unique_reasons = list(set(all_junk_reasons))

    return ClipAnalysis(
        source_path=source_path,
        proxy_path=proxy_path,
        duration_seconds=duration,
        frame_analyses=frame_analyses,
        decision=decision,
        confidence=confidence,
        junk_reasons=unique_reasons,
        suggested_in_point=in_point,
        suggested_out_point=out_point,
        mean_audio_db=mean_audio_db,
    )


def _make_decision(
    frame_analyses: list[FrameAnalysis],
    duration: float,
    mean_audio_db: float | None,
) -> tuple[ClipDecision, ConfidenceLevel, float | None, float | None]:
    """Make a decision about a clip based on frame analyses.

    Decision logic:
    - If 3+ frames at start are junk: Set In_Point to 3.0s
    - If 3+ frames at end are junk: Set Out_Point to End-3s
    - If >50% of clip is junk: Flag entire clip as "Rejected"

    Args:
        frame_analyses: List of frame analysis results.
        duration: Clip duration in seconds.
        mean_audio_db: Mean audio level.

    Returns:
        Tuple of (decision, confidence, suggested_in_point, suggested_out_point).
    """
    if not frame_analyses:
        return ClipDecision.KEEP, ConfidenceLevel.LOW, None, None

    # Count junk frames
    junk_count = sum(1 for fa in frame_analyses if fa.is_junk)
    junk_ratio = junk_count / len(frame_analyses)

    # Count junk at start and end
    start_frames = [fa for fa in frame_analyses if fa.timestamp_seconds <= 3]
    end_frames = [fa for fa in frame_analyses if fa.timestamp_seconds >= duration - 3]

    start_junk = sum(1 for fa in start_frames if fa.is_junk)
    end_junk = sum(1 for fa in end_frames if fa.is_junk)

    in_point = None
    out_point = None

    # Determine confidence based on agreement
    has_opencv = any(fa.blur_score is not None for fa in frame_analyses)
    has_vlm = any(fa.vlm_response is not None for fa in frame_analyses)

    if has_opencv and has_vlm:
        confidence = ConfidenceLevel.HIGH
    elif has_vlm:
        confidence = ConfidenceLevel.MEDIUM
    else:
        confidence = ConfidenceLevel.LOW

    # If >50% junk, reject entire clip
    if junk_ratio > 0.5:
        return ClipDecision.REJECT, confidence, None, None

    # If most start frames are junk, suggest trimming
    if start_junk >= 3 or (len(start_frames) > 0 and start_junk / len(start_frames) > 0.5):
        in_point = 3.0

    # If most end frames are junk, suggest trimming
    if end_junk >= 3 or (len(end_frames) > 0 and end_junk / len(end_frames) > 0.5):
        out_point = max(0, duration - 3.0)

    # Check for very low audio
    if mean_audio_db is not None and mean_audio_db < -50:
        # Very quiet - likely accidental
        confidence = ConfidenceLevel.LOW
        return ClipDecision.REVIEW, confidence, in_point, out_point

    # If any trimming suggested, mark for review
    if in_point is not None or out_point is not None:
        return ClipDecision.REVIEW, confidence, in_point, out_point

    # If any junk detected but not enough to reject/trim, mark for review
    if junk_count > 0:
        return ClipDecision.REVIEW, confidence, in_point, out_point

    return ClipDecision.KEEP, confidence, None, None


def analyze_clips_batch(
    clips: list[tuple[Path, Path | None]],
    use_vlm: bool = True,
    model_name: str = "qwen2.5-vl:7b",
) -> list[ClipAnalysis]:
    """Analyze a batch of video clips.

    Args:
        clips: List of (source_path, proxy_path) tuples.
        use_vlm: Whether to use VLM for analysis.
        model_name: Ollama model name.

    Returns:
        List of ClipAnalysis results.
    """
    results = []

    for i, (source_path, proxy_path) in enumerate(clips):
        logger.info(f"Analyzing clip {i + 1}/{len(clips)}: {source_path.name}")
        result = analyze_clip(source_path, proxy_path, use_vlm, model_name)
        results.append(result)

    # Summary logging
    kept = sum(1 for r in results if r.decision == ClipDecision.KEEP)
    rejected = sum(1 for r in results if r.decision == ClipDecision.REJECT)
    review = sum(1 for r in results if r.decision == ClipDecision.REVIEW)

    logger.info(f"Analysis complete: {kept} keep, {rejected} reject, {review} review")

    return results

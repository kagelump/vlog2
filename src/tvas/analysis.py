"""Stage 3: AI Analysis (Junk Detection)

This module handles AI-powered junk detection using Qwen3 VL (8B) via mlx-vlm,
using native video processing for start/end point detection.
"""

import json
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.video_generate import process_vision_info

from tvas.proxy import get_video_duration

logger = logging.getLogger(__name__)

# Default model for mlx-vlm
DEFAULT_VLM_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-8bit"

# Video sampling parameters
VIDEO_FPS = 10.0  # Sample at 10fps for analysis
VIDEO_SEGMENT_DURATION = 5.0  # Analyze first/last 5 seconds

# Cache for loaded models to avoid reloading
# Structure: {model_name: (model, processor)}
_model_cache: dict[str, tuple] = {}
_model_cache_lock = threading.Lock()


class JunkReason(Enum):
    """Reasons a clip may be flagged as junk."""

    BLUR = "blur"
    DARKNESS = "darkness"
    LENS_CAP = "lens_cap"
    GROUND = "pointing_at_ground"
    ACCIDENTAL = "accidental_trigger"
    MULTIPLE = "multiple_issues"


class ConfidenceLevel(Enum):
    """Confidence level of junk detection."""

    HIGH = "high"  # VLM with clear reasoning
    MEDIUM = "medium"  # VLM with uncertain reasoning
    LOW = "low"  # VLM failed or unclear


class ClipDecision(Enum):
    """Decision for a clip."""

    KEEP = "keep"
    REJECT = "reject"
    REVIEW = "review"  # Uncertain, needs user review


@dataclass
class ClipAnalysis:
    """Complete analysis result for a video clip."""

    source_path: Path
    proxy_path: Path | None
    duration_seconds: float
    decision: ClipDecision
    confidence: ConfidenceLevel
    junk_reasons: list[JunkReason]
    suggested_in_point: float | None = None
    suggested_out_point: float | None = None
    vlm_response: str | None = None
    vlm_summary: str | None = None


def check_model_available(model_name: str = DEFAULT_VLM_MODEL) -> bool:
    """Check if the specified model can be loaded.

    Note: mlx-vlm models are downloaded automatically on first use
    from HuggingFace.

    Args:
        model_name: Name of the mlx-vlm model.

    Returns:
        True (model will be downloaded if needed).
    """
    return True


def _get_or_load_model(model_name: str = DEFAULT_VLM_MODEL):
    """Get a cached model or load it.

    Args:
        model_name: Name of the mlx-vlm model.

    Returns:
        Tuple of (model, processor).
    """
    with _model_cache_lock:
        if model_name not in _model_cache:
            try:
                logger.info(f"Loading mlx-vlm model: {model_name}")
                model, processor = load(model_name)
                _model_cache[model_name] = (model, processor)
                logger.info(f"Model {model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                return None, None

        return _model_cache[model_name]


def analyze_video_segment(
    video_path: Path,
    model_name: str = DEFAULT_VLM_MODEL,
    fps: float = VIDEO_FPS,
    max_pixels: int = 224 * 224,
) -> dict:
    """Analyze a video segment using VLM for junk detection and trim suggestions.

    Args:
        video_path: Path to the video file (can be proxy or original).
        model_name: Name of the mlx-vlm model to use.
        fps: Frames per second to sample from video.
        max_pixels: Maximum pixels for each frame.

    Returns:
        Dictionary with VLM analysis results including trim suggestions.
    """
    model, processor = _get_or_load_model(model_name)
    if model is None or processor is None:
        raise RuntimeError(f"Failed to load model: {model_name}")

    # Get video duration to determine segment strategy
    duration = get_video_duration(video_path) or 0
    
    # Determine analysis strategy
    if duration <= 15:
        # Short video: analyze entire clip
        prompt = f"""Analyze this {duration:.1f}s video clip for quality issues.
Look for: blur/motion blur, camera pointing at ground, lens cap, extremely dark frames, accidental triggers.

Based on your analysis, provide:
1. Whether any part should be trimmed (is_junk: true/false)
2. Specific issues found (issues: list)
3. Recommended start time in seconds (trim_start: number or null if no trim needed at start)
4. Recommended end time in seconds (trim_end: number or null if no trim needed at end)

Respond with ONLY valid JSON:
{{"is_junk": true/false, "reason": "brief explanation", "issues": ["list"], "trim_start": null or number, "trim_end": null or number}}

Be conservative - only suggest trimming if there are clear quality issues."""
    else:
        # Long video: focus on first/last 5 seconds
        prompt = f"""Analyze the beginning and end of this {duration:.1f}s video clip.
Look for quality issues in the FIRST and LAST few seconds: blur, camera pointing at ground, lens cap, dark frames, accidental triggers.

Based on your analysis, provide:
1. Whether the start needs trimming (check first ~5 seconds)
2. Whether the end needs trimming (check last ~5 seconds)
3. Recommended start time in seconds if start has issues (trim_start: number or null)
4. Recommended end time in seconds if end has issues (trim_end: number or null)

Respond with ONLY valid JSON:
{{"is_junk": true/false, "reason": "brief explanation", "issues": ["list"], "trim_start": null or number, "trim_end": null or number}}

Example: If first 3 seconds are junk, set trim_start to 3.0. If last 4 seconds are junk and video is 60s, set trim_end to 56.0."""

    try:
        # Prepare video input for mlx-vlm
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                        "fps": fps,
                        "max_pixels": max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process with mlx-vlm
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, True)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        input_ids = mx.array(inputs["input_ids"])
        pixel_values = mx.array(
            inputs.get("pixel_values_videos", inputs.get("pixel_values"))
        )
        mask = mx.array(inputs["attention_mask"])

        # Prepare kwargs for generation
        gen_kwargs = {}
        if inputs.get("video_grid_thw") is not None:
            gen_kwargs["video_grid_thw"] = mx.array(inputs["video_grid_thw"])
        if inputs.get("image_grid_thw") is not None:
            gen_kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])

        # Generate response
        from mlx_vlm.generate import generate as mlx_generate
        
        output = mlx_generate(
            model,
            processor,
            input_ids,  # type: ignore[arg-type]
            pixel_values,  # type: ignore[arg-type]
            max_tokens=512,
            verbose=False,
            mask=mask,
            **gen_kwargs,
        )

        # Extract text from GenerationResult
        response_text = str(output)

        # Try to parse JSON from response
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                data = json.loads(json_str)
                return {
                    "is_junk": data.get("is_junk", False),
                    "reason": data.get("reason", ""),
                    "issues": data.get("issues", []),
                    "trim_start": data.get("trim_start"),
                    "trim_end": data.get("trim_end"),
                    "raw_response": response_text,
                }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from VLM response: {response_text[:200]}")

        # Fallback: simple keyword detection
        is_junk = any(
            word in response_text.lower()
            for word in ["junk", "blur", "dark", "ground", "accidental", "trim"]
        )
        return {
            "is_junk": is_junk,
            "reason": response_text[:200],
            "issues": [],
            "trim_start": None,
            "trim_end": None,
            "raw_response": response_text,
        }

    except Exception as e:
        logger.error(f"VLM analysis failed for {video_path}: {e}")
        raise


def analyze_clip(
    source_path: Path,
    proxy_path: Path | None = None,
    use_vlm: bool = True,
    model_name: str = DEFAULT_VLM_MODEL,
) -> ClipAnalysis:
    """Perform complete analysis of a video clip using VLM.

    Args:
        source_path: Path to the original video file.
        proxy_path: Path to the AI proxy (preferred for analysis).
        use_vlm: Whether to use VLM for analysis (must be True).
        model_name: mlx-vlm model name for VLM analysis.

    Returns:
        ClipAnalysis with complete analysis results.
    """
    if not use_vlm:
        logger.warning("VLM is required for analysis - enabling it")
        use_vlm = True

    video_to_analyze = proxy_path if proxy_path and proxy_path.exists() else source_path
    duration = get_video_duration(video_to_analyze) or 0

    # Analyze video with VLM
    vlm_result = analyze_video_segment(video_to_analyze, model_name)

    # Parse junk reasons from issues
    junk_reasons: list[JunkReason] = []
    for issue in vlm_result.get("issues", []):
        issue_lower = issue.lower()
        if "blur" in issue_lower:
            junk_reasons.append(JunkReason.BLUR)
        elif "dark" in issue_lower:
            junk_reasons.append(JunkReason.DARKNESS)
        elif "ground" in issue_lower:
            junk_reasons.append(JunkReason.GROUND)
        elif "lens" in issue_lower or "cap" in issue_lower:
            junk_reasons.append(JunkReason.LENS_CAP)
        elif "accidental" in issue_lower:
            junk_reasons.append(JunkReason.ACCIDENTAL)

    # If multiple issues, add that reason
    if len(junk_reasons) > 1:
        junk_reasons.append(JunkReason.MULTIPLE)

    # Determine trim points from VLM suggestions
    trim_start = vlm_result.get("trim_start")
    trim_end = vlm_result.get("trim_end")

    # Determine decision based on VLM output
    is_junk = vlm_result.get("is_junk", False)
    
    if is_junk and (trim_start is None and trim_end is None):
        # Entire clip flagged as junk
        decision = ClipDecision.REJECT
        confidence = ConfidenceLevel.HIGH
    elif trim_start is not None or trim_end is not None:
        # Partial trim suggested
        decision = ClipDecision.REVIEW
        confidence = ConfidenceLevel.HIGH if junk_reasons else ConfidenceLevel.MEDIUM
    elif is_junk:
        # Marked as junk but no specific trim points
        decision = ClipDecision.REVIEW
        confidence = ConfidenceLevel.MEDIUM
    else:
        # Clean clip
        decision = ClipDecision.KEEP
        confidence = ConfidenceLevel.HIGH

    # Deduplicate reasons
    unique_reasons = list(set(junk_reasons))

    return ClipAnalysis(
        source_path=source_path,
        proxy_path=proxy_path,
        duration_seconds=duration,
        decision=decision,
        confidence=confidence,
        junk_reasons=unique_reasons,
        suggested_in_point=trim_start,
        suggested_out_point=trim_end,
        vlm_response=vlm_result.get("raw_response"),
        vlm_summary=vlm_result.get("reason"),
    )


def analyze_clips_batch(
    clips: list[tuple[Path, Path | None]],
    use_vlm: bool = True,
    model_name: str = DEFAULT_VLM_MODEL,
) -> list[ClipAnalysis]:
    """Analyze a batch of video clips.

    Args:
        clips: List of (source_path, proxy_path) tuples.
        use_vlm: Whether to use VLM for analysis (must be True).
        model_name: mlx-vlm model name.

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

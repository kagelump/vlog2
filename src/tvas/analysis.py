"""Stage 3: AI Analysis (Trim Detection)

This module handles AI-powered trim detection using VLMClient (local or API).
"""

from collections.abc import Sequence
import json
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
from typing import Any, Tuple, cast, Optional
from pydantic import BaseModel, ValidationError

from shared.proxy import get_video_duration
from shared import load_prompt, DEFAULT_VLM_MODEL
from shared.vlm_client import VLMClient

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

# Video sampling parameters
VIDEO_FPS = 10.0  # Sample at 10fps for analysis
VIDEO_SEGMENT_DURATION = 5.0  # Analyze first/last 5 seconds

VIDEO_ANALYSIS_PROMPT = load_prompt("video_analysis.txt")

class ConfidenceLevel(Enum):
    """Confidence level of trim detection."""

    HIGH = "high"  # VLM with clear reasoning
    MEDIUM = "medium"  # VLM with uncertain reasoning
    LOW = "low"  # VLM failed or unclear

class DescribeOutput(BaseModel):
    """Structured output from the video description model."""
    start_sec: float | None
    end_sec: float | None
    trim_reason: str  | None
    shot_type: str
    shot_tags: list[str]
    subject_keywords: list[str]
    action_keywords: list[str]
    clip_description: str
    clip_name: str

@dataclass
class ClipAnalysis:
    """Complete analysis result for a video clip."""

    source_path: Path
    proxy_path: Path | None
    duration_seconds: float
    confidence: ConfidenceLevel
    clip_name: str | None = None
    suggested_in_point: float | None = None
    suggested_out_point: float | None = None
    vlm_response: str | None = None
    vlm_summary: str | None = None
    subject_keywords: list[str] | None = None
    action_keywords: list[str] | None = None
    timestamp: float = 0.0

def validate_model_output(parsed: Any) -> dict:
    """Validate parsed JSON from the model against DescribeOutput.

    Returns the dictified model if valid, otherwise raises ValidationError.
    """
    # Use Pydantic v2 API (model_validate + model_dump) for forward compatibility
    obj = DescribeOutput.model_validate(parsed)
    return obj.model_dump()

def analyze_video_segment(
    video_path: Path,
    model_name: str = DEFAULT_VLM_MODEL,
    fps: float = VIDEO_FPS,
    max_pixels: int = 224 * 224,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_preferences: Optional[str] = None,
) -> dict:
    """Analyze a video segment using VLM for trim suggestions.

    Args:
        video_path: Path to the video file (can be proxy or original).
        model_name: Name of the mlx-vlm model to use.
        fps: Frames per second to sample from video.
        max_pixels: Maximum pixels for each frame.
        api_base: Optional API base URL.
        api_key: Optional API key.
        provider_preferences: Optional provider preferences.

    Returns:
        Dictionary with VLM analysis results including trim suggestions.
    """
    client = VLMClient(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        provider_preferences=provider_preferences
    )

    # Adjust FPS to avoid GPU timeouts on long videos (only for local model)
    if not api_base:
        duration = get_video_duration(video_path) or 0
        MAX_FRAMES = 128 # Limit total frames to prevent GPU timeout
        
        if duration > 0:
            calculated_frames = duration * fps
            while calculated_frames > MAX_FRAMES:
                new_fps = fps / 2 
                logger.info(f"Video too long ({duration:.1f}s), adjusting FPS from {fps} to {new_fps:.2f} to keep frames under {MAX_FRAMES}")
                fps = new_fps
                calculated_frames = duration * fps
    else:
        # For API, we might want lower FPS to save tokens/bandwidth, or let the client handle it.
        # The client has a fallback, but let's stick to the passed FPS for now.
        pass

    response = client.generate_from_video(
        prompt=VIDEO_ANALYSIS_PROMPT,
        video_path=video_path,
        fps=fps,
        max_pixels=max_pixels
    )

    if not response or not response.text:
        raise Exception("VLM returned no response")

    try:
        # Clean up response text (remove markdown code blocks if present)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        parsed = json.loads(text)
    except Exception:
        raise Exception(f'Could not deserialize {response.text}')

    # validate and return structured output
    try:
        return validate_model_output(parsed)
    except ValidationError as e:
        # return raw parsed JSON but surface validation error in message
        raise Exception(f'Output did not match expected schema: {e}\nRaw: {parsed}')


def analyze_clip(
    source_path: Path,
    proxy_path: Path | None = None,
    use_vlm: bool = True,
    model_name: str = DEFAULT_VLM_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_preferences: Optional[str] = None,
) -> ClipAnalysis:
    """Perform complete analysis of a video clip using VLM for trim detection.

    Args:
        source_path: Path to the original video file.
        proxy_path: Path to the AI proxy (preferred for analysis).
        use_vlm: Whether to use VLM for analysis (must be True).
        model_name: mlx-vlm model name for VLM analysis.
        api_base: Optional API base URL.
        api_key: Optional API key.
        provider_preferences: Optional provider preferences.

    Returns:
        ClipAnalysis with complete analysis results.
    """
    if not use_vlm:
        logger.warning("VLM is required for analysis - enabling it")
        use_vlm = True

    video_to_analyze = proxy_path if proxy_path and proxy_path.exists() else source_path
    duration = get_video_duration(video_to_analyze) or 0

    # Check for cached analysis
    json_path = video_to_analyze.with_suffix(".json")
    vlm_result = None
    
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                vlm_result = json.load(f)
            logger.info(f"Using cached analysis for {video_to_analyze.name}")
        except Exception as e:
            logger.warning(f"Failed to load cached analysis from {json_path}: {e}")

    if vlm_result is None:
        # Analyze video with VLM
        start_time = time.time()
        try:
            vlm_result = analyze_video_segment(
                video_to_analyze,
                model_name,
                api_base=api_base,
                api_key=api_key,
                provider_preferences=provider_preferences
            )
            elapsed_time = time.time() - start_time
            logger.info(
                f"VLM result for {video_to_analyze.name} ({duration}s) took {elapsed_time:.2f} seconds:\n"
                 f"  In/Out: {vlm_result.get('start_sec')}s to {vlm_result.get('end_sec')}s\n"
                 f"  Reason: {vlm_result.get('trim_reason')}\n"
                 f"  Shot Type/Tags: {vlm_result.get('shot_type')} ({', '.join(vlm_result.get('shot_tags', []))})\n"
                 f"  Subjects: {', '.join(vlm_result.get('subject_keywords', []))}\n"
                 f"  Actions: {', '.join(vlm_result.get('action_keywords', []))}\n"
                 f"  Description: {vlm_result.get('clip_description')}\n"
                 f"  Name: {vlm_result.get('clip_name')}")
            
            # Save result to JSON
            try:
                with open(json_path, "w") as f:
                    json.dump(vlm_result, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save analysis to {json_path}: {e}")
        except Exception as e:
            logger.error(f"Analysis failed for {video_to_analyze.name}: {e}")
            # Return a dummy result or re-raise?
            # For now, let's return a failed analysis
            return ClipAnalysis(
                source_path=source_path,
                proxy_path=proxy_path,
                duration_seconds=duration,
                confidence=ConfidenceLevel.LOW,
                vlm_summary=f"Analysis failed: {e}",
                timestamp=source_path.stat().st_mtime,
            )

    # Extract clip name from VLM suggestions
    clip_name = vlm_result.get("clip_name")
    
    # Determine trim points from VLM suggestions
    trim_start = vlm_result.get("start_sec") # Note: DescribeOutput uses start_sec/end_sec
    trim_end = vlm_result.get("end_sec")

    # Determine confidence based on whether VLM provided suggestions
    if trim_start is not None or trim_end is not None:
        confidence = ConfidenceLevel.HIGH
    else:
        confidence = ConfidenceLevel.MEDIUM

    return ClipAnalysis(
        source_path=source_path,
        proxy_path=proxy_path,
        duration_seconds=duration,
        confidence=confidence,
        clip_name=clip_name,
        suggested_in_point=trim_start,
        suggested_out_point=trim_end,
        vlm_response=vlm_result.get("clip_description"),
        vlm_summary=vlm_result.get("trim_reason"),
        subject_keywords=vlm_result.get("subject_keywords", []),
        action_keywords=vlm_result.get("action_keywords", []),
        timestamp=source_path.stat().st_mtime,
    )


def analyze_clips_batch(
    clips: list[tuple[Path, Path | None]],
    use_vlm: bool = True,
    model_name: str = DEFAULT_VLM_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_preferences: Optional[str] = None,
) -> list[ClipAnalysis]:
    """Analyze a batch of video clips.

    Args:
        clips: List of (source_path, proxy_path) tuples.
        use_vlm: Whether to use VLM for analysis (must be True).
        model_name: mlx-vlm model name.
        api_base: Optional API base URL.
        api_key: Optional API key.
        provider_preferences: Optional provider preferences.

    Returns:
        List of ClipAnalysis results.
    """
    results = []

    for i, (source_path, proxy_path) in enumerate(clips):
        logger.info(f"Processing clip {i + 1}/{len(clips)}: {source_path.name}")
        result = analyze_clip(
            source_path,
            proxy_path,
            use_vlm,
            model_name,
            api_base=api_base,
            api_key=api_key,
            provider_preferences=provider_preferences
        )
        results.append(result)

    # Summary logging
    with_trim = sum(1 for r in results if r.suggested_in_point or r.suggested_out_point)
    logger.info(f"Analysis complete: {len(results)} clips analyzed, {with_trim} with trim suggestions")

    return results

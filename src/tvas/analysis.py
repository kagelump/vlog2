"""Stage 3: AI Analysis (Trim Detection)

This module handles AI-powered trim detection using Qwen3 VL (8B) via mlx-vlm,
using native video processing for start/end point detection.
"""

from collections.abc import Sequence
import json
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
from typing import Any, Tuple, cast
import mlx.core as mx
from pydantic import BaseModel, ValidationError


import torch
from transformers.video_utils import load_video
from transformers import AutoVideoProcessor

from mlx_vlm.video_generate import process_vision_info
from mlx_vlm.video_generate import generate
from mlx_vlm import generate as mlx_vlm_generate
from mlx_vlm import load

from tvas.proxy import get_video_duration

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

# Default model for mlx-vlm
DEFAULT_VLM_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-8bit"

# Video sampling parameters
VIDEO_FPS = 10.0  # Sample at 10fps for analysis
VIDEO_SEGMENT_DURATION = 5.0  # Analyze first/last 5 seconds

TRIM_DETECTION_PROMPT = """
Analyze the video and suggest appropriate start and end points
to remove any unnecessary content (eg. blurry, reframing, etc) at the beginning or end of the clip.
Provide your suggestions in seconds along with a brief explanation.

Determine the primary shot type - based on standard film and editing terminology — choose one of the following:

pov: The camera represents the perspective of a character or subject.
insert: A close-up or detailed shot of an object, text, or small action that provides specific information.
establishing: A wide or introductory shot that sets the scene or location context.
Add descriptive tags that apply to the shot. Choose all that fit from the following list:

static: The camera does not move.
dynamic: The camera moves (pans, tilts, tracks, zooms, etc.).
closeup: Tight framing around a person's face or an object.
medium: Frames the subject roughly from the waist up.
wide: Shows the subject and significant background context.

Describe this video clip using 1-2 sentences. Then give it a short name to use as a filename. Use snake_case.

Respond in JSON format with the following fields:
- start_sec: Suggested start time in seconds (float) or null if no trim needed.
- end_sec: Suggested end time in seconds (float) or null if no trim needed.
- trim_reason: A brief explanation of your trim suggestions.
- shot_type: The primary shot type (pov, insert, establishing).
- shot_tags: A list of descriptive tags (static, dynamic, closeup, medium, wide).
- clip_description: A short summary of the clip's content.
- clip_name: A concise name (2-5 words) for the clip based on its content in lower_snake_case.
"""

# Cache for loaded models to avoid reloading
# Structure: {model_name: (model, processor)}
_model_cache: dict[str, tuple] = {}
_model_cache_lock = threading.Lock()

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
    timestamp: float = 0.0


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


def validate_model_output(parsed: Any) -> dict:
    """Validate parsed JSON from the model against DescribeOutput.

    Returns the dictified model if valid, otherwise raises ValidationError.
    """
    # Use Pydantic v2 API (model_validate + model_dump) for forward compatibility
    obj = DescribeOutput.model_validate(parsed)
    return obj.model_dump()


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


def describe_video(
    model,
    processor,
    video_path: str,
    prompt: str | None = None,
    fps: float = 1.0,
    max_pixels: int = 224 * 224,
    **generate_kwargs
) -> dict:
    """
    Describe a single video using mlx-vlm.

    Args:
        model: The loaded MLX-VLM model
        processor: The loaded processor
        video_path: Path to the video file
        prompt: Optional prompt template. If None, uses default template.
        fps: Frames per second to sample from video
        subtitle: Optional subtitle text to include in the prompt
        max_pixels: Maximum pixel size for frames
        **generate_kwargs: Additional arguments passed to generate()

    Returns:
        dict: Validated description output as dictionary

    Raises:
        Exception: If the model output cannot be parsed or validated
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    _, video_inputs = cast(Tuple[Sequence[Any], Sequence[Any]], process_vision_info(messages))

    inputs = processor(
        text=[text],
        images=None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        video_metadata={'fps': fps, 'total_num_frames': video_inputs[0].shape[0]}
    )

    input_ids = mx.array(inputs["input_ids"])
    mask = mx.array(inputs["attention_mask"])
    video_grid_thw = mx.array(inputs["video_grid_thw"])

    # include kwargs for video layout grid info
    extra = {"video_grid_thw": video_grid_thw}

    pixel_values = inputs.get(
        "pixel_values_videos", inputs.get("pixel_values", None)
    )
    logger.info(f"describe_video: fps: {fps}, total_frame: {video_inputs[0].shape[0]}, pixel_values type={type(pixel_values)} shape={getattr(pixel_values, 'shape', None)}")
    if pixel_values is None:
        raise ValueError("Please provide a valid video or image input.")
    pixel_values = mx.array(pixel_values)

    response = generate(
        model=model,
        processor=processor,
        prompt=text,
        input_ids=input_ids,
        pixel_values=pixel_values,
        mask=mask,
        **extra,
        **generate_kwargs,
    )

    try:
        parsed = json.loads(response.text)
    except Exception:
        raise Exception(f'Could not deserialize {response.text}')

    # validate and return structured output
    try:
        return validate_model_output(parsed)
    except ValidationError as e:
        # return raw parsed JSON but surface validation error in message
        raise Exception(f'Output did not match expected schema: {e}\nRaw: {parsed}')

def analyze_video_segment(
    video_path: Path,
    model_name: str = DEFAULT_VLM_MODEL,
    fps: float = VIDEO_FPS,
    max_pixels: int = 224 * 224,
) -> dict:
    """Analyze a video segment using VLM for trim suggestions.

    Args:
        video_path: Path to the video file (can be proxy or original).
        model_name: Name of the mlx-vlm model to use.
        fps: Frames per second to sample from video.
        max_pixels: Maximum pixels for each frame.

    Returns:
        Dictionary with VLM analysis results including trim suggestions.
    """
    model, processor = _get_or_load_model(model_name)

    # Adjust FPS to avoid GPU timeouts on long videos
    duration = get_video_duration(video_path) or 0
    MAX_FRAMES = 110 # Limit total frames to prevent GPU timeout
    
    if duration > 0:
        calculated_frames = duration * fps
        while calculated_frames > MAX_FRAMES:
            new_fps = fps / 2 
            logger.info(f"Video too long ({duration:.1f}s), adjusting FPS from {fps} to {new_fps:.2f} to keep frames under {MAX_FRAMES}")
            fps = new_fps
            calculated_frames = duration * fps

    return describe_video(
        model,
        processor,
        str(video_path),
        prompt=TRIM_DETECTION_PROMPT,
        fps=fps,
        max_pixels=max_pixels,
    )


def analyze_clip(
    source_path: Path,
    proxy_path: Path | None = None,
    use_vlm: bool = True,
    model_name: str = DEFAULT_VLM_MODEL,
) -> ClipAnalysis:
    """Perform complete analysis of a video clip using VLM for trim detection.

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
        vlm_result = analyze_video_segment(video_to_analyze, model_name)
        elapsed_time = time.time() - start_time
        logger.info(
            f"VLM result for {video_to_analyze.name} ({duration}s) took {elapsed_time:.2f} seconds:\n"
             f"  In/Out: {vlm_result.get('start_sec')}s to {vlm_result.get('end_sec')}s\n"
             f"  Reason: {vlm_result.get('trim_reason')}\n"
             f"  Shot Type/Tags: {vlm_result.get('shot_type')} ({', '.join(vlm_result.get('shot_tags', []))})\n"
             f"  Description: {vlm_result.get('clip_description')}\n"
             f"  Name: {vlm_result.get('clip_name')}")
        
        # Save result to JSON
        try:
            with open(json_path, "w") as f:
                json.dump(vlm_result, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save analysis to {json_path}: {e}")

    # Extract clip name from VLM suggestions
    clip_name = vlm_result.get("clip_name")
    
    # Determine trim points from VLM suggestions
    trim_start = vlm_result.get("trim_start_sec")
    trim_end = vlm_result.get("trim_end_sec")

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
        timestamp=source_path.stat().st_mtime,
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
        logger.info(f"Processing clip {i + 1}/{len(clips)}: {source_path.name}")
        result = analyze_clip(source_path, proxy_path, use_vlm, model_name)
        results.append(result)

    # Summary logging
    with_trim = sum(1 for r in results if r.suggested_in_point or r.suggested_out_point)
    logger.info(f"Analysis complete: {len(results)} clips analyzed, {with_trim} with trim suggestions")

    return results

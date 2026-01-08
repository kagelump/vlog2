"""
Stage 3: AI Analysis (Description)

This module handles AI-powered video description using VLMClient (local or API).
Technical trim detection is handled separately in Stage 6 (trim.py).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from dataclasses import fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, ValidationError
from threading import Lock
from typing import Any, Optional
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time

from shared.proxy import get_video_duration
from shared import load_prompt, DEFAULT_VLM_MODEL
from shared.vlm_client import VLMClient

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

# Video sampling parameters
VIDEO_FPS = 4.0  # Sample at 4fps for analysis
VIDEO_DESCRIBE_PROMPT = load_prompt("video_describe.txt")

# Global lock for thread-safe VLMClient creation (prevents GPU contention in parallel mode)
_vlm_client_lock = Lock()
_cached_vlm_clients: dict[str, VLMClient] = {}

class DescribeOutput(BaseModel):
    """Structured output from the video description model."""
    time_of_day: str | None
    detected_text: str | list[str] | None
    landmark_identification: str | list[str] | None
    environment: str | None
    people_presence: str | None
    mood: str | None
    subject_keywords: list[str]
    action_keywords: list[str]
    clip_description: str
    audio_description: str | None = None
    clip_name: str
    thumbnail_timestamp_sec: float | None = None

class TrimOutput(BaseModel):
    """Structured output from the video trim model."""
    trim_needed: bool
    start_sec: float | None
    end_sec: float | None
    reason: str | None

@dataclass
class ClipAnalysis:
    """Complete analysis result for a video clip."""

    source_path: Path
    proxy_path: Path | None
    duration_seconds: float
    needs_trim: bool = False
    clip_name: str | None = None
    suggested_in_point: float | None = None
    suggested_out_point: float | None = None
    clip_description: str | None = None
    audio_description: str | None = None
    subject_keywords: list[str] | None = None
    action_keywords: list[str] | None = None
    time_of_day: str | None = None
    detected_text: str | list[str] | None = None
    landmark_identification: str | list[str] | None = None
    environment: str | None = None
    people_presence: str | None = None
    mood: str | None = None
    timestamp: float = 0.0
    created_timestamp: str | None = None  # File creation timestamp (YYYY-MM-DD HH:MM:SS)
    modified_timestamp: str | None = None  # File modification timestamp (YYYY-MM-DD HH:MM:SS)
    thumbnail_timestamp_sec: float | None = None
    analysis_duration: float = 0.0  # Time taken to analyze this clip (seconds), not saved to JSON
    beat_id: str | None = None
    beat_title: str | None = None
    beat_classification: str | None = None
    beat_reasoning: str | None = None


def validate_model_output(parsed: Any) -> dict:
    """Validate parsed JSON from the model against DescribeOutput.

    Returns the dictified model if valid, otherwise raises ValidationError.
    """
    # Use Pydantic v2 API (model_validate + model_dump) for forward compatibility
    obj = DescribeOutput.model_validate(parsed)
    return obj.model_dump()


def _get_or_create_vlm_client(
    model_name: str,
    api_base: Optional[str],
    api_key: Optional[str],
    provider_preferences: Optional[str],
    is_parallel: bool = False,
) -> VLMClient:
    """Get or create a VLMClient with thread-safe caching. 
    
    For local models in parallel mode, uses a shared client to avoid
    multiple GPU model loads. For API mode, creates per-thread clients.
    
    Args:
        model_name: Model name.
        api_base: Optional API base URL.
        api_key: Optional API key.
        provider_preferences: Optional provider preferences.
        is_parallel: Whether this is called from parallel mode.
    
    Returns:
        VLMClient instance.
    """
    # API-based models don't have GPU contention, use per-call instances
    if api_base:
        return VLMClient(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            provider_preferences=provider_preferences,
            app_name="tvas (analysis)"
        )
    
    # For local models in parallel mode, use a shared cached client
    if is_parallel:
        cache_key = f"{model_name}:{provider_preferences}"
        with _vlm_client_lock:
            if cache_key not in _cached_vlm_clients:
                logger.info(f"Creating shared VLMClient for {model_name} (parallel mode)")
                _cached_vlm_clients[cache_key] = VLMClient(
                    model_name=model_name,
                    api_base=api_base,
                    api_key=api_key,
                    provider_preferences=provider_preferences,
                    app_name="tvas (analysis)"
                )
            return _cached_vlm_clients[cache_key]
    
    # Sequential mode - create a new client
    return VLMClient(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        provider_preferences=provider_preferences,
        app_name="tvas (analysis)"
    )

def describe_video_segment(
    video_path: Path,
    model_name: str = DEFAULT_VLM_MODEL,
    fps: float = VIDEO_FPS,
    max_pixels: int = 224 * 224,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_preferences: Optional[str] = None,
    is_parallel: bool = False,
    transcription: Optional[str] = None,
) -> dict:
    """Analyze a video segment using VLM for description.

    Args:
        video_path: Path to the video file (can be proxy or original).
        model_name: Name of the mlx-vlm model to use.
        fps: Frames per second to sample from video.
        max_pixels: Maximum pixels for each frame.
        api_base: Optional API base URL.
        api_key: Optional API key.
        provider_preferences: Optional provider preferences.
        is_parallel: Whether called from parallel batch mode.
        transcription: Optional transcription text to include in prompt.

    Returns:
        Dictionary with VLM analysis results.
    """
    client = _get_or_create_vlm_client(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        provider_preferences=provider_preferences,
        is_parallel=is_parallel
    )

    MAX_FRAMES = 500
    if not api_base:
        MAX_FRAMES = 128 # Limit total frames to prevent GPU timeout
    duration = get_video_duration(video_path) or 0
    
    if duration > 0:
        calculated_frames = duration * fps
        while calculated_frames > MAX_FRAMES:
            new_fps = fps / 2 
            logger.info(f"Video too long ({duration:.1f}s), adjusting FPS from {fps} to {new_fps:.2f} to keep frames under {MAX_FRAMES}")
            fps = new_fps
            calculated_frames = duration * fps

    response = client.generate_from_video(
        prompt=VIDEO_DESCRIBE_PROMPT,
        video_path=video_path,
        fps=fps,
        max_pixels=max_pixels,
        transcription=transcription
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


def run_transcribe_subprocess(
    model: str,
    input_path: str,
) -> Optional[str]:
    """Transcribe a single preview file using tvas.transcribe via subprocess.

    Args:
        model: Model ID for mlx_whisper
        input_path: Path to input video/audio file

    Returns:
        Transcription text on success, None if no speech detected or failure.
    """
    try:
        transcribe_script = Path(__file__).parent / "transcribe.py"
        logger.info(f"Running transcription subprocess: {transcribe_script} for {input_path}")
        
        result = subprocess.run(
            [sys.executable, str(transcribe_script), "--model", model, "--input", input_path, "--output", "-"],
            capture_output=True,
            text=True,
            check=False,  # Don't raise on non-zero exit, check returncode manually
        )
        
        if result.returncode == 0:
            # Success
            return result.stdout.strip()
        elif result.returncode == 2:
            # No speech detected - this is a normal condition, not an error
            logger.info(f"No speech detected in {input_path}")
            return None
        else:
            # Other errors
            logger.warning(f"Transcription failed with code {result.returncode}: {result.stdout}\n{result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to run transcription subprocess: {e}")
        return None


def describe_clip(
    source_path: Path,
    proxy_path: Path | None = None,
    use_vlm: bool = True,
    model_name: str = DEFAULT_VLM_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_preferences: Optional[str] = None,
    is_parallel: bool = False,
    clip_index: Optional[int] = None,
    total_clips: Optional[int] = None,
) -> ClipAnalysis:
    """Perform complete description analysis of a video clip using VLM.

    Args:
        source_path: Path to the original video file.
        proxy_path: Path to the AI proxy (preferred for analysis).
        use_vlm: Whether to use VLM for analysis (must be True).
        model_name: mlx-vlm model name for VLM analysis.
        api_base: Optional API base URL.
        api_key: Optional API key.
        provider_preferences: Optional provider preferences.
        is_parallel: Whether this is called from parallel batch mode.
        clip_index: Current clip index for progress tracking (1-indexed).
        total_clips: Total clips in batch for progress tracking.

    Returns:
        ClipAnalysis with complete analysis results (trim fields empty).
    """
    if not use_vlm:
        logger.warning("VLM is required for analysis - enabling it")
        use_vlm = True

    video_to_analyze = proxy_path if proxy_path and proxy_path.exists() else source_path
    duration = get_video_duration(video_to_analyze) or 0

    # Check for cached analysis
    json_path = video_to_analyze.with_suffix(".json")
    vlm_result = None
    analysis_duration = 0.0  # Track time spent on this clip (0 for cached)
    
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                vlm_result = json.load(f)
            logger.info(f"Using cached analysis for {video_to_analyze.name}")
            analysis_duration = 0.0  # Cached results take negligible time
        except Exception as e:
            logger.warning(f"Failed to load cached analysis from {json_path}: {e}")

    if vlm_result is None:
        # Run transcription first
        transcription_text = None
        try:
            transcription_text = run_transcribe_subprocess(model="mlx-community/whisper-large-v3-turbo", input_path=str(video_to_analyze))
            if transcription_text:
                logger.info(f"Generated transcription for {video_to_analyze.name} ({len(transcription_text)} chars)")
        except Exception as e:
            logger.warning(f"Transcription failed for {video_to_analyze.name}: {e}")

        # Analyze video with VLM
        start_time = time.time()
        progress_str = f" [{clip_index}/{total_clips}]" if clip_index and total_clips else ""
        try:
            vlm_result = describe_video_segment(
                video_to_analyze,
                model_name,
                api_base=api_base,
                api_key=api_key,
                provider_preferences=provider_preferences,
                is_parallel=is_parallel,
                transcription=transcription_text
            )
            elapsed_time = time.time() - start_time
            analysis_duration = elapsed_time
            logger.info(
                f"VLM result{progress_str} for {video_to_analyze.name} ({duration}s) took {elapsed_time:.2f} seconds:\n"
                 f"  Time: {vlm_result.get('time_of_day')}\n"
                 f"  Mood: {vlm_result.get('mood')}\n"
                 f"  Environment: {vlm_result.get('environment')}\n"
                 f"  People: {vlm_result.get('people_presence')}\n"
                 f"  Landmark: {vlm_result.get('landmark_identification')}\n"
                 f"  Text: {vlm_result.get('detected_text')}\n"
                 f"  Subjects: {', '.join(vlm_result.get('subject_keywords', []))}\n"
                 f"  Actions: {', '.join(vlm_result.get('action_keywords', []))}\n"
                 f"  Description: {vlm_result.get('clip_description')}\n"
                 f"  Audio: {vlm_result.get('audio_description')}\n"
                 f"  Name: {vlm_result.get('clip_name')}")
            
            # Save result to JSON with metadata
            try:
                # Get file timestamps
                stat_info = source_path.stat()
                created_ts = stat_info.st_birthtime if hasattr(stat_info, 'st_birthtime') else stat_info.st_ctime
                modified_ts = stat_info.st_mtime
                
                json_data = vlm_result.copy()
                json_data["metadata"] = {
                    "duration_seconds": duration,
                    "created_timestamp": datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M:%S"),
                    "modified_timestamp": datetime.fromtimestamp(modified_ts).strftime("%Y-%m-%d %H:%M:%S"),
                    "analysis_timestamp": datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(json_path, "w") as f:
                    json.dump(json_data, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save analysis to {json_path}: {e}")
        except Exception as e:
            logger.error(f"Analysis failed{progress_str} for {video_to_analyze.name}: {e}")
            # Return a failed analysis with proper context preserved
            stat_info = source_path.stat()
            created_ts = stat_info.st_birthtime if hasattr(stat_info, 'st_birthtime') else stat_info.st_ctime
            modified_ts = stat_info.st_mtime
            return ClipAnalysis(
                source_path=source_path,
                proxy_path=proxy_path,
                duration_seconds=duration,
                timestamp=modified_ts,
                created_timestamp=datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M:%S"),
                modified_timestamp=datetime.fromtimestamp(modified_ts).strftime("%Y-%m-%d %H:%M:%S"),
                analysis_duration=0.0,  # Failed analysis
            )

    # Extract clip name from VLM suggestions
    clip_name = vlm_result.get("clip_name")
    
    # Trim detection moved to separate phase
    needs_trim = False
    trim_start = None
    trim_end = None

    # Get file timestamps
    stat_info = source_path.stat()
    created_ts = stat_info.st_birthtime if hasattr(stat_info, 'st_birthtime') else stat_info.st_ctime
    modified_ts = stat_info.st_mtime
    
    return ClipAnalysis(
        source_path=source_path,
        proxy_path=proxy_path,
        duration_seconds=duration,
        needs_trim=needs_trim,
        clip_name=clip_name,
        suggested_in_point=trim_start,
        suggested_out_point=trim_end,
        clip_description=vlm_result.get("clip_description"),
        audio_description=vlm_result.get("audio_description"),
        subject_keywords=vlm_result.get("subject_keywords", []),
        action_keywords=vlm_result.get("action_keywords", []),
        time_of_day=vlm_result.get("time_of_day"),
        detected_text=vlm_result.get("detected_text"),
        landmark_identification=vlm_result.get("landmark_identification"),
        environment=vlm_result.get("environment"),
        people_presence=vlm_result.get("people_presence"),
        mood=vlm_result.get("mood"),
        timestamp=modified_ts,
        created_timestamp=datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M:%S"),
        modified_timestamp=datetime.fromtimestamp(modified_ts).strftime("%Y-%m-%d %H:%M:%S"),
        thumbnail_timestamp_sec=vlm_result.get("thumbnail_timestamp_sec"),
        analysis_duration=analysis_duration,
    )


def aggregate_analysis_csv(project_dir: Path, all_results: list) -> Path:
    """Export aggregated analysis results to CSV.
    
    Args:
        project_dir: Directory to save the CSV.
        all_results: List of dictionaries containing analysis data.
        
    Returns:
        Path to the created CSV file.
    """
    csv_path = project_dir / "analysis.csv"
    
    # Mapping from ClipAnalysis field names to raw JSON keys (DescribeOutput schema)
    # Fields not listed here are assumed to have the same name in both
    # Supports dot notation for nested keys (e.g., "beat.beat_id")
    json_key_map = {
        "needs_trim": "trim",
        "suggested_in_point": "start_sec",
        "suggested_out_point": "end_sec",
        # Metadata fields
        "created_timestamp": "metadata.created_timestamp",
        "modified_timestamp": "metadata.modified_timestamp",
        # Beat fields
        "beat_id": "beat.beat_id",
        "beat_title": "beat.beat_title",
        "beat_classification": "beat.classification",
        "beat_reasoning": "beat.reasoning",
    }

    # Dynamically get headers from ClipAnalysis dataclass
    clip_fields = fields(ClipAnalysis)
    headers = [f.name.replace("_", " ").title() for f in clip_fields]
    
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for data in all_results:
                row = []
                
                for field in clip_fields:
                    # Determine the key to look for in the JSON data
                    key_path = json_key_map.get(field.name, field.name)
                    
                    # Navigate dot notation
                    val = data
                    for key in key_path.split("."):
                        if isinstance(val, dict):
                            val = val.get(key)
                        else:
                            val = None
                            break
                        
                    # Fallback for old metadata location (root -> metadata) if explicit path failed
                    # This preserves backward compatibility if not strictly defined in map
                    if val is None and key_path == field.name and "metadata" in data:
                        val = data["metadata"].get(field.name)
                        
                    # Format specific types
                    if isinstance(val, list):
                        val = ", ".join(str(x) for x in val)
                    elif isinstance(val, float):
                        val = f"{val:.2f}"
                    elif isinstance(val, bool):
                        val = "Yes" if val else "No"
                    elif val is None:
                        val = ""
                    else:
                        val = str(val)
                        
                    row.append(val)
                
                writer.writerow(row)
        
        logger.info(f"Aggregated analysis CSV exported to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to export aggregated analysis CSV: {e}")
        
    return csv_path


def aggregate_analysis_json(project_dir: Path) -> Path:
    """Combine all individual JSON analysis files into a single sorted analysis.json.
    
    Args:
        project_dir: Directory containing individual .json analysis files (e.g. proxy dir).
        
    Returns:
        Path to the combined analysis.json.
    """
    logger.info(f"Aggregating analysis results in {project_dir}")
    all_results = []
    
    # Find all .json files that have a 'metadata' key (to distinguish from analysis.json itself)
    for json_file in project_dir.glob("*.json"):
        if json_file.name == "analysis.json":
            continue
            
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                if "metadata" in data and "created_timestamp" in data["metadata"]:
                    # Ensure source_path is included if not present (relative to project_dir)
                    if "source_path" not in data:
                        data["source_path"] = str(json_file.with_suffix(".mp4"))
                    all_results.append(data)
        except Exception as e:
            logger.warning(f"Failed to read {json_file}: {e}")
            
    # Sort by created_timestamp
    all_results.sort(key=lambda x: x.get("metadata", {}).get("created_timestamp", ""))
    
    output_path = project_dir / "analysis.json"
    try:
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Successfully aggregated {len(all_results)} results into {output_path}")
        
        # Also export CSV
        aggregate_analysis_csv(project_dir, all_results)
        
        # Save the path to a magic file for other tools to pick up
        try:
            magic_file = Path.home() / ".tvas_current_analysis"
            magic_file.write_text(str(output_path.resolve()))
            logger.debug(f"Updated {magic_file}")
            
            # Copy Resolve import script to DaVinci Resolve's scripts folder
            resolve_script_src = Path(__file__).parent.parent / "resolve" / "import_timeline.py"
            if resolve_script_src.exists():
                resolve_script_dest_dir = Path.home() / "Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Comp"
                if resolve_script_dest_dir.exists():
                    shutil.copy2(resolve_script_src, resolve_script_dest_dir / "import_timeline.py")
                    logger.debug(f"Copied Resolve script to {resolve_script_dest_dir}")
        except Exception as e:
            logger.warning(f"Failed to update magic file or copy Resolve script: {e}")
        
    except Exception as e:
        logger.error(f"Failed to write aggregated analysis.json: {e}")
        
    return output_path


def analyze_clips_batch(
    clips: list[tuple[Path, Path | None]],
    use_vlm: bool = True,
    model_name: str = DEFAULT_VLM_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_preferences: Optional[str] = None,
    max_workers: int = 1,
    progress_callback: Optional[Any] = None,
) -> list[ClipAnalysis]:
    """Analyze a batch of video clips.

    Args:
        clips: List of (source_path, proxy_path) tuples.
        use_vlm: Whether to use VLM for analysis (must be True).
        model_name: mlx-vlm model name.
        api_base: Optional API base URL.
        api_key: Optional API key.
        provider_preferences: Optional provider preferences.
        max_workers: Number of parallel workers (default 1 for sequential).
        progress_callback: Optional callback(current, total, result).

    Returns:
        List of ClipAnalysis results.
    """
    # Validate and clamp max_workers
    original_workers = max_workers
    max_workers = max(1, min(max_workers, 16))
    if original_workers != max_workers:
        logger.info(f"Adjusted workers from {original_workers} to {max_workers}")
    
    if max_workers > 1 and api_base is None:
        logger.warning(
            f"Parallel analysis with {max_workers} workers on local model may cause GPU memory issues. "
            f"Consider using API mode (--openrouter or --lmstudio) for parallel processing."
        )
    if max_workers <= 1:
        results = []
        for i, (source_path, proxy_path) in enumerate(clips):
            logger.info(f"Processing clip {i + 1}/{len(clips)}: {source_path.name}")
            result = describe_clip(
                source_path,
                proxy_path,
                use_vlm,
                model_name,
                api_base=api_base,
                api_key=api_key,
                provider_preferences=provider_preferences,
                is_parallel=False,
            )
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(clips), result)
    else:
        logger.info(f"Starting parallel analysis with {max_workers} workers on {len(clips)} clips")
        results_map = {}
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_clip = {
                executor.submit(
                    describe_clip,
                    source_path,
                    proxy_path,
                    use_vlm,
                    model_name,
                    api_base=api_base,
                    api_key=api_key,
                    provider_preferences=provider_preferences,
                    is_parallel=True,
                    clip_index=i + 1,
                    total_clips=len(clips),
                ): (i, source_path, proxy_path)
                for i, (source_path, proxy_path) in enumerate(clips)
            }
            
            for future in as_completed(future_to_clip):
                index, source_path, proxy_path = future_to_clip[future]
                completed_count += 1
                try:
                    result = future.result()
                    results_map[index] = result
                    if progress_callback:
                        progress_callback(completed_count, len(clips), result)
                    logger.debug(f"Finished clip {index + 1}/{len(clips)}: {source_path.name}")
                except Exception as e:
                    logger.error(f"Clip {index + 1}/{len(clips)} ({source_path.name}) failed: {e}")
                    # Create a failed result preserving original context
                    stat_info = source_path.stat()
                    created_ts = stat_info.st_birthtime if hasattr(stat_info, 'st_birthtime') else stat_info.st_ctime
                    modified_ts = stat_info.st_mtime
                    failed_result = ClipAnalysis(
                        source_path=source_path,
                        proxy_path=proxy_path,
                        duration_seconds=0,
                        timestamp=modified_ts,
                        created_timestamp=datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M:%S"),
                        modified_timestamp=datetime.fromtimestamp(modified_ts).strftime("%Y-%m-%d %H:%M:%S"),
                        analysis_duration=0.0,  # Failed analysis
                    )
                    results_map[index] = failed_result
                    if progress_callback:
                        progress_callback(completed_count, len(clips), failed_result)
                
                if completed_count % max(1, len(clips) // 10) == 0 or completed_count == len(clips):
                    logger.info(f"Progress: {completed_count}/{len(clips)} clips analyzed")
        
        # Sort results back into original order
        results = [results_map[i] for i in range(len(clips))]

    # Aggregate results into analysis.json if we have proxies
    if clips:
        # Determine project directory from the first proxy path or source path
        first_source, first_proxy = clips[0]
        project_dir = (first_proxy or first_source).parent
        aggregate_analysis_json(project_dir)

    # Summary logging
    logger.info(f"Analysis complete: {len(results)} clips analyzed")

    return results
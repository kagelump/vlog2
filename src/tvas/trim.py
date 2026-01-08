"""Stage 6: Trim Detection

This module handles technical trim detection using VLM on the start/end segments of clips.
"""

import json
import logging
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Any

from shared import DEFAULT_VLM_MODEL, load_prompt
from shared.vlm_client import VLMClient
from shared.proxy import get_video_duration
from shared.ffmpeg_utils import detect_best_video_codec, check_ffmpeg_available
from tvas.analysis import ClipAnalysis, aggregate_analysis_json

logger = logging.getLogger(__name__)

VIDEO_TRIM_PROMPT = load_prompt("video_trim.txt")
TRIM_CONTEXT_SECONDS = 5.0

def generate_trim_proxy(video_path: Path) -> Path | None:
    """Generate a temporary proxy containing only the start and end segments.
    
    Returns:
        Path to the temporary file, or None if failed/unnecessary.
    """
    if not check_ffmpeg_available():
        logger.error("FFmpeg not available for trim proxy generation")
        return None

    duration = get_video_duration(video_path)
    if not duration or duration < (TRIM_CONTEXT_SECONDS * 2) + 1:
        # Clip is short enough, just use original
        return video_path

    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = Path(temp_file.name)
    temp_file.close()

    try:
        # Construct filter_complex to concat start and end
        # Use a slightly shorter duration to avoid potential EOF issues with trim
        safe_duration = duration - 0.1
        
        filter_complex = (
            f"[0:v]trim=0:{TRIM_CONTEXT_SECONDS},setpts=PTS-STARTPTS[v0];"
            f"[0:v]trim={safe_duration-TRIM_CONTEXT_SECONDS}:{safe_duration},setpts=PTS-STARTPTS[v1];"
            f"[v0][v1]concat=n=2:v=1[outv]"
        )
        
        # We explicitly disable audio (-an) to avoid complexity with audio stream matching
        codec_flags = detect_best_video_codec()
        
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video_path),
            "-filter_complex", filter_complex,
            "-map", "[outv]"
        ] + codec_flags + [
            "-an", 
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to generate trim proxy for {video_path}: {e}")
        if e.stderr:
            logger.warning(f"FFmpeg stderr: {e.stderr}")
        if e.stdout:
            logger.warning(f"FFmpeg stdout: {e.stdout}")
        if output_path.exists():
            output_path.unlink()
        return None
    except Exception as e:
        logger.warning(f"Failed to generate trim proxy for {video_path}: {e}")
        if output_path.exists():
            output_path.unlink()
        return None

def detect_trim_for_file(
    json_path: Path,
    client: VLMClient,
) -> bool:
    """Detect trim points for a single clip sidecar file.
    
    Returns: True if processed, False if skipped.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {json_path}: {e}")
        return False

    # Check if trim already exists
    if "trim" in data and isinstance(data["trim"], dict):
        logger.info(f"Skipping trim for {json_path.name} (already exists)")
        return False

    # Check classification
    # Data structure: data['beat']['classification'] or top level?
    # align_beats updates "beat" object.
    classification = None
    if "beat" in data and isinstance(data["beat"], dict):
        classification = data["beat"].get("classification")
    
    if classification in ["REMOVE", "WEAK"]:
        logger.info(f"Skipping trim for {json_path.name} ({classification})")
        return False

    # Determine video path
    source_path_str = data.get("source_path")
    proxy_path_str = data.get("proxy_path")
    
    video_path = None
    if proxy_path_str:
        proxy_path = Path(proxy_path_str)
        if not proxy_path.is_absolute():
            proxy_path = json_path.parent / proxy_path
        if proxy_path.exists():
            video_path = proxy_path
    
    if not video_path and source_path_str:
        source_path = Path(source_path_str)
        if not source_path.is_absolute():
            source_path = json_path.parent / source_path
        if source_path.exists():
            video_path = source_path
    
    # Fallback: use JSON stem + .mp4
    if not video_path:
        default_path = json_path.parent / f"{json_path.stem}.mp4"
        if default_path.exists():
            video_path = default_path
    
    if not video_path:
        logger.warning(f"No video file found for {json_path.name}")
        return False

    trim_proxy = generate_trim_proxy(video_path)
    used_proxy = trim_proxy and trim_proxy != video_path
    target_path = trim_proxy if trim_proxy else video_path
    
    try:
        response = client.generate_from_video(
            prompt=VIDEO_TRIM_PROMPT,
            video_path=target_path,
            fps=1.0, 
            max_pixels=224*224
        )
        
        if response and response.text:
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
                
            try:
                result = json.loads(text)
                trim_needed = result.get("trim_needed", False)
                
                # Update data with nested trim object
                trim_data = {
                    "trim_needed": trim_needed,
                    "suggested_in_point": None,
                    "suggested_out_point": None,
                    "reason": None
                }
                
                if trim_needed:
                    start_rel = result.get("start_sec")
                    end_rel = result.get("end_sec")
                    reason = result.get("reason")
                    
                    duration = data.get("metadata", {}).get("duration_seconds") or get_video_duration(video_path) or 0
                    
                    final_start = 0.0
                    final_end = duration
                    
                    if start_rel is not None:
                        # Logic: if start_rel < TRIM_CONTEXT_SECONDS, it's in the first segment
                        if start_rel <= TRIM_CONTEXT_SECONDS:
                            final_start = start_rel
                        # If > context, it's weird, likely near start?
                        
                    if end_rel is not None:
                        if used_proxy:
                            if end_rel >= TRIM_CONTEXT_SECONDS:
                                # In second segment
                                offset = end_rel - TRIM_CONTEXT_SECONDS
                                final_end = (duration - TRIM_CONTEXT_SECONDS) + offset
                            else:
                                # In first segment? 
                                final_end = end_rel
                        else:
                            final_end = end_rel
                    
                    # Safety clamps
                    final_start = max(0.0, final_start)
                    final_end = min(duration, final_end)
                    if final_start >= final_end:
                        final_start = 0.0
                        final_end = duration
                        trim_data["trim_needed"] = False # Invalidate
                    else:
                        trim_data["suggested_in_point"] = final_start
                        trim_data["suggested_out_point"] = final_end
                        trim_data["reason"] = reason
                        
                    logger.info(f"Trim detected for {json_path.name}: {final_start:.1f}-{final_end:.1f}")
                else:
                    logger.info(f"No trim needed for {json_path.name}")
                
                # Save nested object
                data["trim"] = trim_data
                
                # Save
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)
                    
            except json.JSONDecodeError:
                logger.error(f"Failed to parse trim response for {json_path.name}")
    except Exception as e:
        logger.error(f"Trim processing failed for {json_path.name}: {e}")
    finally:
        if used_proxy and trim_proxy.exists():
            trim_proxy.unlink()
            
    return True

def detect_trims_batch(
    project_dir: Path,
    model_name: str = DEFAULT_VLM_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_preferences: Optional[str] = None,
    max_workers: int = 1,
) -> None:
    """Run trim detection on all analyzed clips in project_dir."""
    
    json_files = sorted(
        [f for f in project_dir.glob("*.json") if f.name != "analysis.json"],
        key=lambda x: x.name
    )
    
    if not json_files:
        logger.warning("No clips found for trim detection")
        return

    logger.info(f"Starting trim detection for {len(json_files)} clips...")
    
    # Initialize VLM Client
    # Thread-safety is handled by VLMClient internals usually, but we need per-thread clients if using API?
    # _get_or_create_vlm_client in analysis.py handled this.
    # Here we can just instantiate.
    # VLMClient supports API base.
    
    # For local models, max_workers should be 1.
    if not api_base and max_workers > 1:
        logger.warning("Forcing max_workers=1 for local model trim detection")
        max_workers = 1
        
    client = VLMClient(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        provider_preferences=provider_preferences,
        app_name="tvas (trim)"
    )
    
    processed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # If API, we need separate clients? VLMClient is not strictly thread safe if it holds state?
        # Actually VLMClient holds `self.model`.
        # Local model: cannot share across threads easily for inference unless locked.
        # API: stateless mostly.
        # analysis.py used `_get_or_create_vlm_client`.
        # I'll rely on sequential for local, parallel for API.
        
        futures = []
        for json_path in json_files:
            if max_workers > 1 and api_base:
                # Create new client for each? Or share?
                # VLMClient with API is thread safe (urllib).
                futures.append(executor.submit(detect_trim_for_file, json_path, client))
            else:
                futures.append(executor.submit(detect_trim_for_file, json_path, client))
                
        for future in as_completed(futures):
            if future.result():
                processed_count += 1
                
    logger.info(f"Trim detection complete. Processed {processed_count}/{len(json_files)} eligible clips.")
    
    aggregate_analysis_json(project_dir)

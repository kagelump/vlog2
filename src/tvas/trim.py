"""Stage 6: Trim Detection

This module handles technical trim detection using VLM on the start/end segments of clips.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Any

from shared import DEFAULT_VLM_MODEL, load_prompt
from shared.vlm_client import VLMClient
from shared.clip_formatter import format_clip_for_prompt
from tvas.analysis import aggregate_analysis_json

logger = logging.getLogger(__name__)

VIDEO_TRIM_PROMPT = load_prompt("video_trim.txt")

def detect_trim_for_file(
    json_path: Path,
    client: VLMClient,
    outline_text: Optional[str] = None,
) -> bool:
    """Detect trim points and best moments for a single clip sidecar file.
    
    Returns: True if processed, False if skipped.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {json_path}: {e}")
        return False

    # Check if trim already exists and has the new best_moment field
    if "trim" in data and isinstance(data["trim"], dict) and "best_moment" in data["trim"]:
        logger.info(f"Skipping trim for {json_path.name} (already exists)")
        return False

    # Check classification
    classification = None
    if "beat" in data and isinstance(data["beat"], dict):
        classification = data["beat"].get("classification")
    
    if classification in ["REMOVE", "WEAK", "HERO"]:
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

    # We no longer use generate_trim_proxy because we need to see the full clip
    # for "Best Moment" detection.
    target_path = video_path
    
    # Construct Prompt with Context
    # 1. Base Prompt
    # 2. Outline (Global Context)
    # 3. Clip Analysis (Local Context)
    
    prompt = f"{VIDEO_TRIM_PROMPT}\n\n"
    
    if outline_text:
        prompt += f"--- STORY OUTLINE ---\n{outline_text}\n\n"
        
    # Add clip analysis context using shared formatter for consistency
    prompt += f"--- CLIP ANALYSIS ---\n{format_clip_for_prompt(data, include_technical=True)}\n"

    try:
        response = client.generate_from_video(
            prompt=prompt,
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
                
                tech_trim = result.get("technical_trim", {})
                best_moment = result.get("best_moment", {})
                
                trim_needed = tech_trim.get("trim_needed", False)
                start_sec = tech_trim.get("start_sec")
                end_sec = tech_trim.get("end_sec")
                reason = tech_trim.get("reason")
                
                # Update data with comprehensive trim object
                trim_data = {
                    "trim_needed": trim_needed,
                    "suggested_in_point": start_sec,
                    "suggested_out_point": end_sec,
                    "reason": reason,
                    "technical_trim": tech_trim,
                    "best_moment": best_moment,
                    "action_peaks": result.get("action_peaks", []),
                    "dead_zones": result.get("dead_zones", [])
                }
                
                if trim_needed:
                    logger.info(f"Trim detected for {json_path.name}: {start_sec}-{end_sec}")
                
                if best_moment:
                    bs = best_moment.get("start_sec")
                    be = best_moment.get("end_sec")
                    score = best_moment.get("score")
                    logger.info(f"Best moment for {json_path.name}: {bs}-{be} (score: {score})")

                # Save nested object
                data["trim"] = trim_data
                
                # Save
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)
                    
            except json.JSONDecodeError:
                logger.error(f"Failed to parse trim response for {json_path.name}")
    except Exception as e:
        logger.error(f"Trim processing failed for {json_path.name}: {e}")
            
    return True

def detect_trims_batch(
    project_dir: Path,
    outline_path: Optional[Path] = None,
    model_name: str = DEFAULT_VLM_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_preferences: Optional[str] = None,
    max_workers: int = 1,
    progress_callback: Optional[callable] = None,
) -> None:
    """Run trim detection on all analyzed clips in project_dir.
    
    Args:
        progress_callback: Optional callback(current, total, clip_name) called after each clip.
    """
    
    json_files = sorted(
        [f for f in project_dir.glob("*.json") if f.name != "analysis.json"],
        key=lambda x: x.name
    )
    
    if not json_files:
        logger.warning("No clips found for trim detection")
        return

    logger.info(f"Starting trim detection for {len(json_files)} clips...")
    
    # Load Outline if available
    outline_text = None
    if outline_path and outline_path.exists():
        try:
            outline_text = outline_path.read_text()
            logger.info(f"Loaded story outline from {outline_path}")
        except Exception as e:
            logger.warning(f"Failed to read outline from {outline_path}: {e}")
    elif outline_path:
        logger.warning(f"Outline file not found: {outline_path}")
    
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
    total_count = len(json_files)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # If API, we need separate clients? VLMClient is not strictly thread safe if it holds state?
        # Actually VLMClient holds `self.model`.
        # Local model: cannot share across threads easily for inference unless locked.
        # API: stateless mostly.
        # analysis.py used `_get_or_create_vlm_client`.
        # I'll rely on sequential for local, parallel for API.
        
        futures = {}
        for json_path in json_files:
            if max_workers > 1 and api_base:
                # Create new client for each? Or share?
                # VLMClient with API is thread safe (urllib).
                futures[executor.submit(detect_trim_for_file, json_path, client, outline_text)] = json_path
            else:
                futures[executor.submit(detect_trim_for_file, json_path, client, outline_text)] = json_path
                
        for future in as_completed(futures):
            json_path = futures[future]
            if future.result():
                processed_count += 1
            
            # Report progress
            completed = len([f for f in futures if f.done()])
            if progress_callback:
                progress_callback(completed, total_count, json_path.name)
                
    logger.info(f"Trim detection complete. Processed {processed_count}/{len(json_files)} eligible clips.")
    
    aggregate_analysis_json(project_dir)

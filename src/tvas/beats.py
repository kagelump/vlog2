"""Stage 5: Beat Alignment

Aligns analyzed clips to story beats defined in an outline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from tvas.analysis import aggregate_analysis_json
from shared import DEFAULT_VLM_MODEL, load_prompt
from shared.vlm_client import VLMClient
from shared.proxy import extract_frame

logger = logging.getLogger(__name__)

# Load beat alignment prompt from shared prompts folder
BEAT_ALIGNMENT_PROMPT = load_prompt("beat_alignment.txt")

def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load JSON {path}: {e}")
        return None

def save_json(path: Path, data: dict[str, Any]) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save JSON {path}: {e}")

def align_beats(
    project_dir: Path,
    outline_path: Path,
    model_name: str = DEFAULT_VLM_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_preferences: Optional[str] = None,
) -> None:
    """Align clips in project_dir to beats in outline_path."""
    
    if not outline_path.exists():
        logger.error(f"Outline file not found: {outline_path}")
        return

    outline_content = outline_path.read_text()
    
    # Initialize VLM Client
    client = VLMClient(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        provider_preferences=provider_preferences,
        app_name="tvas (beat)"
    )

    # Find all JSON sidecars
    # We assume sidecars are named same as video files + .json?
    # Or just *.json excluding analysis.json?
    # "align all clips in a directory with json sidecars"
    candidate_json_files = [f for f in project_dir.glob("*.json") if f.name != "analysis.json"]
    
    if not candidate_json_files:
        logger.warning(f"No clip JSON files found in {project_dir}")
        return

    # Sort by created_timestamp in metadata
    json_with_metadata = []
    for f in candidate_json_files:
        data = load_json(f)
        if data:
            created_ts = data.get("metadata", {}).get("created_timestamp", "")
            json_with_metadata.append((f, data, created_ts))
    
    # Sort by the timestamp string (YYYY-MM-DD HH:MM:SS)
    json_with_metadata.sort(key=lambda x: x[2])
    
    logger.info(f"Found {len(json_with_metadata)} clips to align.")

    previous_clip_data = None

    for i, (json_path, current_data, _) in enumerate(json_with_metadata):
        # Skip if already has beat
        if "beat" in current_data:
            logger.info(f"Skipping {json_path.name} (already has beat)")
            previous_clip_data = current_data
            continue

        logger.info(f"Aligning beat for {json_path.name} ({i+1}/{len(json_with_metadata)})...")

        # Prepare context
        # Handle Thumbnail Generation
        thumbnail_path = json_path.with_suffix(".jpg")
        thumb_timestamp = current_data.get("thumbnail_timestamp_sec")
        
        if not thumbnail_path.exists() and thumb_timestamp is not None:
            # Try to find video file to extract from
            # Prefer proxy_path if available and exists, fallback to source_path
            video_path = None
            proxy_path_str = current_data.get("proxy_path")
            source_path_str = current_data.get("source_path")
            
            if proxy_path_str and Path(proxy_path_str).exists():
                video_path = Path(proxy_path_str)
            elif source_path_str and Path(source_path_str).exists():
                video_path = Path(source_path_str)
            elif json_path.with_suffix(".mp4").exists():
                # Try same name as json but .mp4
                video_path = json_path.with_suffix(".mp4")
                
            if video_path:
                logger.info(f"Generating thumbnail for {json_path.name} at {thumb_timestamp}s")
                extract_frame(video_path, thumb_timestamp, thumbnail_path, max_dimension=512)
            else:
                logger.warning(f"Could not find video file to generate thumbnail for {json_path.name}")

        image_paths = []
        if thumbnail_path.exists():
            image_paths.append(thumbnail_path)
        
        # Construct Prompt
        # Limit previous clip data to avoid context window issues
        prev_summary = None
        if previous_clip_data:
            prev_summary = {
                "clip_name": previous_clip_data.get("clip_name"),
                "description": previous_clip_data.get("clip_description"),
                "time": previous_clip_data.get("metadata", {}).get("created_timestamp"),
                "assigned_beat": previous_clip_data.get("beat", {}).get("beat_id")
            }

        prompt = f"{BEAT_ALIGNMENT_PROMPT}\n\n"
        prompt += f"--- OUTLINE ---\n{outline_content}\n\n"
        
        if prev_summary:
            prompt += f"--- PREVIOUS CLIP ---\n{json.dumps(prev_summary, indent=2)}\n\n"
            
        prompt += f"--- CURRENT CLIP JSON ---\n{json.dumps(current_data, indent=2)}\n"

        # Call VLM
        try:
            response = client.generate(
                prompt=prompt,
                image_paths=image_paths,
                max_tokens=500,
                temperature=0.1 # Low temp for deterministic logic
            )
            
            if response and response.text:
                # Parse JSON from response
                text = response.text.strip()
                # Attempt to extract JSON if wrapped in markdown
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                
                try:
                    beat_data = json.loads(text)
                    current_data["beat"] = beat_data
                    save_json(json_path, current_data)
                    logger.info(f"Assigned {json_path.name} to {beat_data.get('beat_id')}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response for {json_path.name}: {response.text}")
            else:
                logger.error(f"No response from VLM for {json_path.name}")
                
        except Exception as e:
            logger.error(f"Error processing {json_path.name}: {e}")

        previous_clip_data = current_data

    logger.info("Beat alignment complete.")
    
    # Re-aggregate analysis.json and analysis.csv to include beat info
    aggregate_analysis_json(project_dir)

#!/usr/bin/env python3
"""
Import Timeline to DaVinci Resolve

This script reads the most recent TVAS analysis (via ~/.tvas_current_analysis)
and creates a new timeline in the current DaVinci Resolve project with all clips
placed in order. It attempts to respect trim points if available.

Prerequisites:
- DaVinci Resolve Studio must be running.
- Python 3.6+ (Resolve uses its own Python or system Python depending on config).
"""

import sys
import os
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def get_current_analysis_path():
    """Read the path from the magic file."""
    magic_file = Path.home() / ".tvas_current_analysis"
    if not magic_file.exists():
        logger.error(f"Magic file not found at {magic_file}")
        return None
    
    try:
        path_str = magic_file.read_text().strip()
        path = Path(path_str)
        if not path.exists():
            logger.error(f"Analysis file does not exist: {path}")
            return None
        return path
    except Exception as e:
        logger.error(f"Failed to read magic file: {e}")
        return None

def main():
    resolve = app.GetResolve() 
    if not resolve:
        logger.error("Could not connect to DaVinci Resolve. Make sure it is running.")
        sys.exit(1)

    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    
    if not project:
        logger.error("No project is open in DaVinci Resolve.")
        sys.exit(1)

    analysis_path = get_current_analysis_path()
    if not analysis_path:
        sys.exit(1)

    logger.info(f"Loading analysis from: {analysis_path}")
    
    try:
        with open(analysis_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        sys.exit(1)

    # The data structure can be a list (aggregated) or dict (single export)
    # Based on timeline.py: export_analysis_json returns {"clips": [...]}
    # Based on analysis.py: aggregate_analysis_json returns [...]
    if isinstance(data, dict):
        clips_data = data.get("clips", [])
        export_time = data.get("export_time", "")
    else:
        clips_data = data
        export_time = ""

    if not clips_data:
        logger.error("No clips found in analysis.json")
        sys.exit(1)

    media_pool = project.GetMediaPool()
    root_folder = media_pool.GetRootFolder()
    
    # Organize timelines in a separate bin
    try:
        timelines_bin = None
        for folder in root_folder.GetSubFolderList():
            if folder.GetName() == "Timelines":
                timelines_bin = folder
                break
        if not timelines_bin:
            timelines_bin = media_pool.AddSubFolder(root_folder, "Timelines")
            logger.info("Created 'Timelines' bin")
    except Exception as e:
        logger.warning(f"Error creating/finding Timelines bin: {e}")
        timelines_bin = root_folder

    # Create import-specific bin for clips
    import_name = f"TVAS Import {export_time}".strip()
    if not import_name:
        import_name = "TVAS Import"
        
    try:
        target_bin = media_pool.AddSubFolder(root_folder, import_name)
        if not target_bin:
            logger.warning(f"Could not create bin '{import_name}', using root.")
            target_bin = root_folder
        else:
            media_pool.SetCurrentFolder(target_bin)
    except Exception as e:
        logger.warning(f"Error creating bin: {e}")
        target_bin = root_folder

    logger.info("Importing clips...")
    
    # Group clips by beat
    # Use a dict to store lists of clips per beat
    # Use a list to store beat IDs in order of appearance
    beats = {}
    beat_order = []
    
    for clip_info in clips_data:
        source_path_str = clip_info.get("source_path")
        if not source_path_str:
            continue
            
        source_path = Path(source_path_str)
        # Ensure source_path is absolute relative to analysis file if needed
        if not source_path.is_absolute():
            source_path = analysis_path.parent / source_path
            
        if not source_path.exists():
            logger.warning(f"File not found: {source_path}")
            continue
            
        # Import
        imported_items = media_pool.ImportMedia([str(source_path)])
        if not imported_items:
            logger.warning(f"Failed to import: {source_path}")
            continue
            
        item = imported_items[0]
        
        # Set Clip Color based on beat classification
        beat_info = clip_info.get("beat") or {}
        classification = beat_info.get("classification")
        beat_id = beat_info.get("beat_id") or "Unassigned"
        beat_title = beat_info.get("beat_title") or ""
        
        color_map = {
            "HERO": "Pink",
            "HIGHLIGHT": "Green",
            "TRANSITION": "Blue",
            "WEAK": "Tan",
            "REMOVE": "Slate",
        }
        clip_color = color_map.get(classification, "Orange")
        item.SetClipColor(clip_color)
        
        # Set Orientation to 0 degrees
        item.SetClipProperty("Image Orientation", "0")
        
        # Rename clip to clip_name from JSON
        clip_name = clip_info.get("clip_name")
        if clip_name:
            item.SetClipProperty("Clip Name", clip_name)
        
        # Calculate trim
        fps = float(item.GetClipProperty("FPS"))
        if not fps: 
            fps = 60.0
            
        # Get trim info from nested object, fallback to top-level for backward compatibility or if flattened
        trim_info = clip_info.get("trim") or clip_info
        
        start_sec = trim_info.get("suggested_in_point")
        end_sec = trim_info.get("suggested_out_point")
        
        # Duration is usually in metadata or top level
        duration_sec = clip_info.get("duration_seconds", 0)
        
        # Check needs_trim flag
        needs_trim = trim_info.get("trim_needed")
        if needs_trim is None:
             # Fallback to old key
             needs_trim = trim_info.get("needs_trim", False)
        
        if needs_trim and (start_sec is not None or end_sec is not None):
            start_frame = int((start_sec or 0.0) * fps)
            total_frames = int(item.GetClipProperty("Frames") or (duration_sec * fps))
            end_frame = int(end_sec * fps) if end_sec is not None else total_frames
            
            # Clamp
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)
            
            if start_frame < end_frame:
                append_entry = {
                    "mediaPoolItem": item,
                    "startFrame": start_frame,
                    "endFrame": end_frame - 1
                }
        
        # Add to beat group
        if beat_id not in beats:
            beats[beat_id] = {
                "title": beat_title,
                "clips": []
            }
            beat_order.append(beat_id)
        else:
            # Consolidate titles: choose the shorter non-empty title to ensure consistency
            current_title = beats[beat_id]["title"]
            if beat_title and beat_title.strip():
                if not current_title or len(beat_title) < len(current_title):
                    beats[beat_id]["title"] = beat_title
        
        beats[beat_id]["clips"].append(append_entry)

    if not beats:
        logger.error("No clips available to add to timeline.")
        sys.exit(1)

    # Create Timelines
    try:
        # Set current folder to Timelines bin
        media_pool.SetCurrentFolder(timelines_bin)
        
        # Create a folder for this import's timelines if using multiple
        import_timelines_bin = media_pool.AddSubFolder(timelines_bin, import_name)
        if import_timelines_bin:
             media_pool.SetCurrentFolder(import_timelines_bin)
        
        for beat_id in beat_order:
            beat_data = beats[beat_id]
            beat_title = beat_data["title"]
            clips = beat_data["clips"]
            
            # Sanitize name
            safe_title = "".join(c for c in beat_title if c.isalnum() or c in (' ', '_', '-')).strip()
            timeline_name = f"{beat_id} - {safe_title}" if safe_title else beat_id
            
            logger.info(f"Creating timeline: {timeline_name} ({len(clips)} clips)")
            
            timeline = media_pool.CreateEmptyTimeline(timeline_name)
            if not timeline:
                logger.error(f"Failed to create timeline {timeline_name}")
                continue
                
            media_pool.AppendToTimeline(clips)
        
        logger.info("Done!")
        
    except Exception as e:
        logger.error(f"Error creating timelines: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
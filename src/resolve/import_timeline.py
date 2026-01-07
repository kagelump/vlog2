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

def load_resolve():
    """Attempt to load the DaVinci Resolve scripting module."""
    try:
        import DaVinciResolveScript as dvr_script
        return dvr_script.scriptapp("Resolve")
    except ImportError:
        # Try default paths for macOS
        expected_path = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules/"
        if os.path.exists(expected_path):
            sys.path.append(expected_path)
            try:
                import DaVinciResolveScript as dvr_script
                return dvr_script.scriptapp("Resolve")
            except ImportError:
                pass
    return None

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

    # Sort clips? The analysis might already be sorted, but let's ensure chronological order via timestamp if present
    # Or trust the list order (which tvas beats/analysis logic usually sorts).
    # tvas analysis sorts by created_timestamp in aggregation.
    
    media_pool = project.GetMediaPool()
    root_folder = media_pool.GetRootFolder()
    
    try:
        target_bin = media_pool.AddSubFolder(root_folder, "Clips")
        if not target_bin:
            # Maybe it exists? Or failed. Fallback to root.
            logger.warning(f"Could not create bin 'Clips', using root.")
            target_bin = root_folder
        else:
            media_pool.SetCurrentFolder(target_bin)
    except Exception as e:
        logger.warning(f"Error creating bin: {e}")
        target_bin = root_folder
    
    logger.info("Importing clips...")
    
    append_list = []
    
    for i, clip_info in enumerate(data):
        source_path = clip_info.get("source_path")
        if not source_path:
            logger.warning(f"No source_path in clip info")
            continue
        
        # Ensure source_path is absolute
        source_path = Path(source_path)
        if not source_path.is_absolute():
            source_path = analysis_path.parent / source_path
        source_path = str(source_path)
        
        if not os.path.exists(source_path):
            logger.warning(f"File not found: {source_path}")
            continue
            
        # Import
        imported_items = media_pool.ImportMedia([source_path])
        if not imported_items:
            logger.warning(f"Failed to import: {source_path}")
            continue
            
        item = imported_items[0]
        
        # Calculate trim
        # Resolve API works in Frames.
        fps = float(item.GetClipProperty("FPS"))
        
        # Fallback FPS if 0 or None (images?)
        if not fps: 
            fps = 60.0 # Standard fallback or project FPS
            
        start_sec = clip_info.get("suggested_in_point")
        end_sec = clip_info.get("suggested_out_point")
        duration_sec = clip_info.get("duration_seconds", 0)
        
        # Check if trim is actually needed
        needs_trim = clip_info.get("needs_trim", False)
        
        if needs_trim and (start_sec is not None or end_sec is not None):
            start_frame = int((start_sec or 0.0) * fps)
            # If end_sec is None, use full duration
            # item.GetClipProperty("Frames") might be string
            total_frames = int(item.GetClipProperty("Frames") or (duration_sec * fps))
            
            end_frame = int(end_sec * fps) if end_sec is not None else total_frames
            
            # Clamp
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)
            
            if start_frame >= end_frame:
                logger.warning(f"Invalid trim for {clip_info.get('clip_name')}: {start_frame}-{end_frame}. Using full clip.")
                append_list.append(item)
            else:
                # Add dictionary for AppendToTimeline
                append_entry = {
                    "mediaPoolItem": item,
                    "startFrame": start_frame,
                    "endFrame": end_frame - 1 # Resolve uses inclusive? usually 0-based index or frame count? 
                                              # AppendToTimeline dict usually takes startFrame (source) and endFrame (source).
                }
                append_list.append(append_entry)
        else:
            append_list.append(item)

    if not append_list:
        logger.error("No clips available to add to timeline.")
        sys.exit(1)

    # Create Timeline
    timeline_name = f"TVAS_Timeline_{import_name}"
    logger.info(f"Creating timeline: {timeline_name}")
    
    # CreateEmptyTimeline is usually available on MediaPool
    try:
        timeline = media_pool.CreateEmptyTimeline(timeline_name)
        if not timeline:
            logger.error("Failed to create timeline.")
            sys.exit(1)
            
        # Append
        logger.info(f"Appending {len(append_list)} clips...")
        media_pool.AppendToTimeline(append_list)
        
        logger.info("Done!")
        
    except Exception as e:
        logger.error(f"Error creating timeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

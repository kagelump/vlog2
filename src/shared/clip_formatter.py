"""Clip Analysis Formatting Utilities

Provides consistent formatting of clip analysis data for prompts and display.
"""

from typing import Any, Dict, List, Union


def format_clip_for_prompt(clip: Dict[str, Any], include_technical: bool = False) -> str:
    """Format a single clip analysis for use in LLM prompts.
    
    Args:
        clip: Clip analysis dictionary
        include_technical: If True, include technical metadata like file paths
        
    Returns:
        Formatted string representation of the clip
    """
    metadata = clip.get('metadata', {})
    lines = []
    
    # Header
    clip_name = clip.get('clip_name') or clip.get('source_path', 'Unknown')
    if isinstance(clip_name, dict):
        clip_name = clip_name.get('name', 'Unknown')
    lines.append(f"Clip: {clip_name}")
    
    # Temporal metadata
    if metadata.get('created_timestamp'):
        lines.append(f"Timestamp: {metadata['created_timestamp']}")
    
    # Duration
    if metadata.get('duration'):
        lines.append(f"Duration: {metadata['duration']}s")
    elif clip.get('duration_seconds'):
        lines.append(f"Duration: {clip['duration_seconds']:.1f}s")
    
    # Contextual information
    if clip.get('time_of_day'):
        lines.append(f"Time: {clip['time_of_day']}")
    if clip.get('environment'):
        lines.append(f"Environment: {clip['environment']}")
    if clip.get('people_presence'):
        lines.append(f"People: {clip['people_presence']}")
    if clip.get('landmark_identification'):
        lines.append(f"Landmarks: {clip['landmark_identification']}")
    if clip.get('detected_text'):
        lines.append(f"Text: {clip['detected_text']}")
    if clip.get('mood'):
        lines.append(f"Mood: {clip['mood']}")
    
    # Descriptions
    if clip.get('clip_description'):
        lines.append(f"Description: {clip['clip_description']}")
    if clip.get('audio_description'):
        lines.append(f"Audio: {clip['audio_description']}")
    
    # Keywords
    if clip.get('subject_keywords'):
        keywords = clip['subject_keywords']
        if isinstance(keywords, list):
            lines.append(f"Subjects: {', '.join(keywords)}")
        else:
            lines.append(f"Subjects: {keywords}")
    
    if clip.get('action_keywords'):
        keywords = clip['action_keywords']
        if isinstance(keywords, list):
            lines.append(f"Actions: {', '.join(keywords)}")
        else:
            lines.append(f"Actions: {keywords}")
    
    # Beat classification (if available)
    if 'beat' in clip and isinstance(clip['beat'], dict):
        beat = clip['beat']
        if beat.get('classification'):
            lines.append(f"Classification: {beat['classification']}")
        if beat.get('story_beat'):
            lines.append(f"Story Beat: {beat['story_beat']}")
        if beat.get('purpose'):
            lines.append(f"Purpose: {beat['purpose']}")
    
    # Technical metadata (optional)
    if include_technical:
        if clip.get('source_path'):
            lines.append(f"Source: {clip['source_path']}")
        if clip.get('proxy_path'):
            lines.append(f"Proxy: {clip['proxy_path']}")
    
    return "\n".join(lines)


def format_clips_for_prompt(
    clips: Union[List[Dict[str, Any]], Dict[str, Any]], 
    separator: str = "\n---\n",
    include_technical: bool = False
) -> str:
    """Format multiple clips for use in LLM prompts.
    
    Args:
        clips: List of clip dictionaries, or analysis.json dict with 'clips' key
        separator: String to use between clips
        include_technical: If True, include technical metadata
        
    Returns:
        Formatted string of all clips
    """
    # Handle both list and dict formats
    if isinstance(clips, dict):
        clips = clips.get('clips', [])
    
    clip_summaries = [format_clip_for_prompt(clip, include_technical) for clip in clips]
    return separator.join(clip_summaries)

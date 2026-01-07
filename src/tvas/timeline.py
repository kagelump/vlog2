"""Stage 4: Timeline Generation

This module generates OpenTimelineIO timelines from analyzed clips
for import into DaVinci Resolve.
"""

import csv
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import opentimelineio as otio

logger = logging.getLogger(__name__)


@dataclass
class TimelineClip:
    """Represents a clip to be added to the timeline."""

    source_path: Path
    name: str
    duration_seconds: float
    in_point_seconds: float = 0.0
    out_point_seconds: float | None = None
    confidence: str = "high"
    camera_source: str = ""
    ai_notes: str = ""


@dataclass
class TimelineConfig:
    """Configuration for timeline generation."""

    name: str = "TVAS Timeline"
    framerate: float = 60.


def create_csv_export(clips: list[TimelineClip], otio_path: Path) -> None:
    """Create a CSV export of the clips alongside the OTIO file."""
    csv_path = otio_path.with_suffix(".csv")
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["File Name", "Clip Directory", "EDL Clip Name"])
            for clip in clips:
                writer.writerow([
                    clip.source_path.name,
                    str(clip.source_path.parent),
                    clip.name,
                ])
        logger.info(f"CSV export created: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to write CSV export: {e}")


def create_timeline(
    clips: list[TimelineClip],
    output_path: Path,
    config: TimelineConfig | None = None,
) -> Path | None:
    """Create an OpenTimelineIO timeline from clips.

    Args:
        clips: List of TimelineClip objects.
        output_path: Path for the output .otio file.
        config: Timeline configuration.

    Returns:
        Path to the created timeline, or None if failed.
    """
    if config is None:
        config = TimelineConfig()

    # Create timeline
    timeline = otio.schema.Timeline(name=config.name)
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)

    for clip_data in clips:
        # Calculate duration and in/out points
        duration = clip_data.duration_seconds
        in_point = clip_data.in_point_seconds
        out_point = clip_data.out_point_seconds or duration

        # Create time range
        rate = config.framerate
        available_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, rate),
            duration=otio.opentime.RationalTime(duration * rate, rate),
        )

        source_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(in_point * rate, rate),
            duration=otio.opentime.RationalTime((out_point - in_point) * rate, rate),
        )

        # Create external reference for the source file
        media_ref = otio.schema.ExternalReference(
            target_url=str(clip_data.source_path.absolute()),
            available_range=available_range,
        )

        # Create clip
        clip = otio.schema.Clip(
            name=clip_data.name,
            media_reference=media_ref,
            source_range=source_range,
        )

        # Add metadata
        clip.metadata["tvas"] = {
            "camera_source": clip_data.camera_source,
            "confidence": clip_data.confidence,
            "ai_notes": clip_data.ai_notes,
            "original_duration": duration,
        }

        track.append(clip)

    timeline.tracks.append(track)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        otio.adapters.write_to_file(timeline, str(output_path))
        logger.info(f"Timeline created: {output_path}")
        create_csv_export(clips, output_path)
        return output_path
    except Exception as e:
        logger.error(f"Failed to write timeline: {e}")
        return None


def create_timeline_from_analysis(
    analyses: list,  # List of ClipAnalysis
    output_path: Path,
    config: TimelineConfig | None = None,
) -> Path | None:
    """Create a timeline from analyzed clips.

    Args:
        analyses: List of ClipAnalysis from the analysis stage.
        output_path: Path for the output file.
        config: Timeline configuration.

    Returns:
        Path to the created timeline.
    """
    from tvas.analysis import ClipAnalysis

    # Sort analyses by timestamp
    analyses.sort(key=lambda x: x.timestamp)

    timeline_clips = []

    for analysis in analyses:
        if not isinstance(analysis, ClipAnalysis):
            continue

        # Build AI notes from VLM summary
        ai_notes = analysis.vlm_summary or ""

        # Use AI-generated clip name if available, otherwise use filename
        clip_name = analysis.clip_name or analysis.source_path.stem

        # Only use suggested trim points if AI says trim is needed
        in_point = analysis.suggested_in_point if analysis.needs_trim else 0.0
        out_point = analysis.suggested_out_point if analysis.needs_trim else None

        timeline_clip = TimelineClip(
            source_path=analysis.source_path,
            name=clip_name,
            duration_seconds=analysis.duration_seconds,
            in_point_seconds=in_point or 0.0,
            out_point_seconds=out_point,
            confidence=analysis.confidence.value,
            camera_source=analysis.source_path.parent.name,
            ai_notes=ai_notes,
        )
        timeline_clips.append(timeline_clip)

    return create_timeline(timeline_clips, output_path, config)


def get_timeline_summary(timeline_path: Path) -> dict | None:
    """Get a summary of an existing timeline.

    Args:
        timeline_path: Path to the .otio file.

    Returns:
        Dictionary with timeline summary, or None if failed.
    """
    try:
        timeline = otio.adapters.read_from_file(str(timeline_path))

        total_clips = 0
        total_duration = 0.0

        for track in timeline.tracks:
            for clip in track:
                if isinstance(clip, otio.schema.Clip):
                    total_clips += 1
                    if clip.source_range:
                        total_duration += clip.source_range.duration.to_seconds()

        return {
            "name": timeline.name,
            "total_clips": total_clips,
            "total_duration_seconds": total_duration,
        }

    except Exception as e:
        logger.error(f"Failed to read timeline: {e}")
        return None


def export_analysis_json(
    analyses: list,
    output_path: Path,
) -> Path:
    """Export analysis results to JSON for debugging/logging.

    Args:
        analyses: List of ClipAnalysis objects.
        output_path: Path for the JSON file.

    Returns:
        Path to the created JSON file.
    """
    from tvas.analysis import ClipAnalysis

    data = {
        "export_time": datetime.now().isoformat(),
        "total_clips": len(analyses),
        "clips": [],
    }

    for analysis in analyses:
        if not isinstance(analysis, ClipAnalysis):
            continue

        clip_data = {
            "source_path": str(analysis.source_path),
            "proxy_path": str(analysis.proxy_path) if analysis.proxy_path else None,
            "duration_seconds": analysis.duration_seconds,
            "confidence": analysis.confidence.value,
            "needs_trim": analysis.needs_trim,
            "clip_name": analysis.clip_name,
            "suggested_in_point": analysis.suggested_in_point,
            "suggested_out_point": analysis.suggested_out_point,
            "vlm_response": analysis.vlm_response,
            "vlm_summary": analysis.vlm_summary,
            "timestamp": analysis.timestamp,
            "metadata": {
                "created_timestamp": analysis.created_timestamp,
                "modified_timestamp": analysis.modified_timestamp,
            },
            "beat": {
                "beat_id": analysis.beat_id,
                "beat_title": analysis.beat_title,
                "classification": analysis.beat_classification,
                "reasoning": analysis.beat_reasoning,
            }
        }
        data["clips"].append(clip_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Analysis exported to {output_path}")
    
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
    
    return output_path

"""Stage 5: Timeline Generation

This module generates OpenTimelineIO timelines from analyzed and reviewed clips
for import into DaVinci Resolve.
"""

import json
import logging
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
    is_rejected: bool = False
    is_uncertain: bool = False
    confidence: str = "high"
    camera_source: str = ""
    ai_notes: str = ""


@dataclass
class TimelineConfig:
    """Configuration for timeline generation."""

    name: str = "TVAS Timeline"
    framerate: float = 24.0
    include_rejected: bool = True  # Include rejected clips (marked with red markers)
    add_markers: bool = True  # Add color markers for AI decisions


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
        # Skip rejected clips if configured
        if clip_data.is_rejected and not config.include_rejected:
            continue

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
            "is_rejected": clip_data.is_rejected,
            "is_uncertain": clip_data.is_uncertain,
            "ai_notes": clip_data.ai_notes,
            "original_duration": duration,
        }

        # Add markers if configured
        if config.add_markers:
            if clip_data.is_rejected:
                # Red marker for rejected
                marker = otio.schema.Marker(
                    name="AI: Rejected",
                    color=otio.schema.MarkerColor.RED,
                    marked_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=otio.opentime.RationalTime(1, rate),
                    ),
                )
                clip.markers.append(marker)
            elif clip_data.is_uncertain:
                # Yellow marker for uncertain
                marker = otio.schema.Marker(
                    name="AI: Review",
                    color=otio.schema.MarkerColor.YELLOW,
                    marked_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=otio.opentime.RationalTime(1, rate),
                    ),
                )
                clip.markers.append(marker)
            # Green clips don't need markers (they're good)

        track.append(clip)

    timeline.tracks.append(track)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        otio.adapters.write_to_file(timeline, str(output_path))
        logger.info(f"Timeline created: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to write timeline: {e}")
        return None


def create_timeline_from_analysis(
    analyses: list,  # List of ClipAnalysis
    reviewed_clips: list | None,  # List of ClipReviewItem with user decisions
    output_path: Path,
    config: TimelineConfig | None = None,
) -> Path | None:
    """Create a timeline from analyzed clips, incorporating user reviews.

    Args:
        analyses: List of ClipAnalysis from the analysis stage.
        reviewed_clips: Optional list of reviewed clips with user decisions.
        output_path: Path for the output file.
        config: Timeline configuration.

    Returns:
        Path to the created timeline.
    """
    from tvas.analysis import ClipAnalysis
    from tvas.review_ui import ClipReviewItem, UserDecision

    timeline_clips = []

    # Create lookup for user decisions
    user_decisions = {}
    if reviewed_clips:
        for rc in reviewed_clips:
            if isinstance(rc, ClipReviewItem):
                user_decisions[str(rc.source_path)] = rc

    for analysis in analyses:
        if not isinstance(analysis, ClipAnalysis):
            continue

        # Check for user override
        user_review = user_decisions.get(str(analysis.source_path))

        # Determine final decision based on user review
        is_rejected = False
        is_uncertain = False

        if user_review:
            if user_review.user_decision == UserDecision.REJECT:
                is_rejected = True
            elif user_review.user_decision == UserDecision.KEEP:
                is_rejected = False
                is_uncertain = False
            else:
                # Undecided - mark as uncertain
                is_uncertain = True
        else:
            # No user review - mark as uncertain for review
            is_uncertain = True

        # Build AI notes from VLM summary
        ai_notes = analysis.vlm_summary or ""

        # Use AI-generated clip name if available, otherwise use filename
        clip_name = analysis.clip_name or analysis.source_path.stem

        timeline_clip = TimelineClip(
            source_path=analysis.source_path,
            name=clip_name,
            duration_seconds=analysis.duration_seconds,
            in_point_seconds=analysis.suggested_in_point or 0.0,
            out_point_seconds=analysis.suggested_out_point,
            is_rejected=is_rejected,
            is_uncertain=is_uncertain,
            confidence=analysis.confidence.value,
            camera_source=analysis.source_path.parent.name,
            ai_notes=ai_notes,
        )
        timeline_clips.append(timeline_clip)

    return create_timeline(timeline_clips, output_path, config)


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
            "clip_name": analysis.clip_name,
            "suggested_in_point": analysis.suggested_in_point,
            "suggested_out_point": analysis.suggested_out_point,
            "vlm_response": analysis.vlm_response,
            "vlm_summary": analysis.vlm_summary,
        }
        data["clips"].append(clip_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Analysis exported to {output_path}")
    return output_path


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
        rejected_count = 0
        uncertain_count = 0

        for track in timeline.tracks:
            for clip in track:
                if isinstance(clip, otio.schema.Clip):
                    total_clips += 1
                    if clip.source_range:
                        total_duration += clip.source_range.duration.to_seconds()

                    # Check metadata for AI status
                    tvas_meta = clip.metadata.get("tvas", {})
                    if tvas_meta.get("is_rejected"):
                        rejected_count += 1
                    elif tvas_meta.get("is_uncertain"):
                        uncertain_count += 1

        return {
            "name": timeline.name,
            "total_clips": total_clips,
            "total_duration_seconds": total_duration,
            "rejected_clips": rejected_count,
            "uncertain_clips": uncertain_count,
            "kept_clips": total_clips - rejected_count - uncertain_count,
        }

    except Exception as e:
        logger.error(f"Failed to read timeline: {e}")
        return None

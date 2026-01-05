"""Tests for the timeline module."""

from pathlib import Path
import pytest

from tvas.timeline import (
    TimelineClip,
    TimelineConfig,
)


class TestTimelineConfig:
    """Tests for TimelineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TimelineConfig()

        assert config.name == "TVAS Timeline"
        assert config.framerate == 60.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TimelineConfig(
            name="Tokyo Vlog",
            framerate=30.0,
        )

        assert config.name == "Tokyo Vlog"
        assert config.framerate == 30.0


class TestTimelineClip:
    """Tests for TimelineClip dataclass."""

    def test_timeline_clip_defaults(self):
        """Test TimelineClip with default values."""
        clip = TimelineClip(
            source_path=Path("/videos/test.mp4"),
            name="test",
            duration_seconds=10.0,
        )

        assert clip.in_point_seconds == 0.0
        assert clip.out_point_seconds is None
        assert clip.confidence == "high"
        assert clip.camera_source == ""
        assert clip.ai_notes == ""

    def test_timeline_clip_custom(self):
        """Test TimelineClip with custom values."""
        clip = TimelineClip(
            source_path=Path("/videos/test.mp4"),
            name="test",
            duration_seconds=10.0,
            in_point_seconds=2.0,
            out_point_seconds=8.0,
            confidence="medium",
            camera_source="DJIPocket3",
            ai_notes="Trim suggestion at start",
        )

        assert clip.in_point_seconds == 2.0
        assert clip.out_point_seconds == 8.0
        assert clip.confidence == "medium"
        assert clip.camera_source == "DJIPocket3"


# Note: Integration tests for create_timeline and create_timeline_from_analysis
# would require actual video files and ClipAnalysis objects, so they are better
# suited for integration/E2E test suites rather than unit tests.

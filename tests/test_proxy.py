"""Tests for the proxy module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared.proxy import (
    ProxyConfig,
    ProxyType,
    build_ai_proxy_command,
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


class TestBuildAIProxyCommand:
    """Tests for AI proxy command building."""

    def test_build_ai_proxy_command_defaults(self):
        """Test building AI proxy command with defaults."""
        source = Path("/videos/test.mp4")
        output = Path("/cache/test_ai_proxy.mp4")
        config = ProxyConfig(use_hardware_accel=False)

        cmd = build_ai_proxy_command(source, output, config)

        assert "ffmpeg" in cmd
        assert "-i" in cmd
        assert str(source) in cmd
        assert str(output) in cmd
        assert "-an" in cmd  # No audio
        assert "-vf" in cmd

        # Check video filter
        vf_index = cmd.index("-vf")
        vf_value = cmd[vf_index + 1]
        assert "scale=640" in vf_value
        assert "fps=10" in vf_value

    def test_build_ai_proxy_command_custom_config(self):
        """Test building AI proxy command with custom config."""
        source = Path("/videos/test.mp4")
        output = Path("/cache/test_ai_proxy.mp4")
        config = ProxyConfig(
            width=480,
            framerate=5,
            bitrate="300k",
            use_hardware_accel=False,
        )

        cmd = build_ai_proxy_command(source, output, config)

        vf_index = cmd.index("-vf")
        vf_value = cmd[vf_index + 1]
        assert "scale=480" in vf_value
        assert "fps=5" in vf_value


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
        assert "prores_ks" in cmd  # ProRes codec
        assert "-profile:v" in cmd
        assert "0" in cmd  # ProRes Proxy profile


class TestProxyConfig:
    """Tests for ProxyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProxyConfig()

        assert config.width == 640
        assert config.framerate == 10
        assert config.bitrate == "500k"
        assert config.use_hardware_accel is True
        assert config.batch_size == 5
        assert config.cooldown_seconds == 30
        assert config.max_temp_celsius == 95

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProxyConfig(
            width=320,
            framerate=5,
            bitrate="250k",
        )

        assert config.width == 320
        assert config.framerate == 5
        assert config.bitrate == "250k"


class TestProxyType:
    """Tests for ProxyType enum."""

    def test_proxy_types(self):
        """Test proxy type values."""
        assert ProxyType.AI_PROXY.value == "ai_proxy"
        assert ProxyType.EDIT_PROXY.value == "edit_proxy"

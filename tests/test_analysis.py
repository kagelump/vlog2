"""Tests for the analysis module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared import DEFAULT_VLM_MODEL


class TestDefaultModel:
    """Tests for the default model constant."""

    def test_default_model_is_qwen3_vl_8b(self):
        """Test that the default model is Qwen3 VL 8B."""
        assert DEFAULT_VLM_MODEL == "mlx-community/Qwen3-VL-8B-Instruct-8bit"


# Note: Integration tests for analyze_video_segment, analyze_clip, and
# analyze_clips_batch would require actual video files and the mlx-vlm
# model loaded, so they are better suited for integration/E2E test suites
# rather than unit tests.

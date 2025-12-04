"""Tests for the analysis module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tvas.analysis import (
    ClipDecision,
    ConfidenceLevel,
    DEFAULT_VLM_MODEL,
    JunkReason,
    check_model_available,
)


class TestModelAvailability:
    """Tests for model availability check."""

    def test_check_model_available(self):
        """Test that model availability always returns True (models auto-download)."""
        assert check_model_available() is True
        assert check_model_available("mlx-community/Qwen3-VL-8B-Instruct-8bit") is True


class TestDefaultModel:
    """Tests for the default model constant."""

    def test_default_model_is_qwen3_vl_8b(self):
        """Test that the default model is Qwen3 VL 8B."""
        assert DEFAULT_VLM_MODEL == "mlx-community/Qwen3-VL-8B-Instruct-8bit"


class TestJunkReason:
    """Tests for JunkReason enum."""

    def test_junk_reason_values(self):
        """Test that JunkReason enum has expected values."""
        assert JunkReason.BLUR.value == "blur"
        assert JunkReason.DARKNESS.value == "darkness"
        assert JunkReason.LENS_CAP.value == "lens_cap"
        assert JunkReason.GROUND.value == "pointing_at_ground"
        assert JunkReason.ACCIDENTAL.value == "accidental_trigger"
        assert JunkReason.MULTIPLE.value == "multiple_issues"


class TestClipDecision:
    """Tests for ClipDecision enum."""

    def test_clip_decision_values(self):
        """Test that ClipDecision enum has expected values."""
        assert ClipDecision.KEEP.value == "keep"
        assert ClipDecision.REJECT.value == "reject"
        assert ClipDecision.REVIEW.value == "review"


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_confidence_level_values(self):
        """Test that ConfidenceLevel enum has expected values."""
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"


# Note: Integration tests for analyze_video_segment, analyze_clip, and
# analyze_clips_batch would require actual video files and the mlx-vlm
# model loaded, so they are better suited for integration/E2E test suites
# rather than unit tests.

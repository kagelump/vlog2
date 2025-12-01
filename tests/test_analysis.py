"""Tests for the analysis module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tvas.analysis import (
    ClipDecision,
    ConfidenceLevel,
    DEFAULT_VLM_MODEL,
    FrameAnalysis,
    JunkReason,
    _make_decision,
    check_mlx_vlm_available,
)


class TestMlxVlmAvailability:
    """Tests for mlx-vlm availability check."""

    def test_check_mlx_vlm_available_present(self):
        """Test when mlx-vlm is available."""
        import tvas.analysis
        original = tvas.analysis.MLX_VLM_AVAILABLE
        tvas.analysis.MLX_VLM_AVAILABLE = True
        try:
            assert tvas.analysis.check_mlx_vlm_available() is True
        finally:
            tvas.analysis.MLX_VLM_AVAILABLE = original

    def test_check_mlx_vlm_available_missing(self):
        """Test when mlx-vlm is not available."""
        import tvas.analysis
        original = tvas.analysis.MLX_VLM_AVAILABLE
        tvas.analysis.MLX_VLM_AVAILABLE = False
        try:
            assert tvas.analysis.check_mlx_vlm_available() is False
        finally:
            tvas.analysis.MLX_VLM_AVAILABLE = original


class TestDefaultModel:
    """Tests for the default model constant."""

    def test_default_model_is_correct(self):
        """Test that the default model is the expected one."""
        assert DEFAULT_VLM_MODEL == "mlx-community/Qwen3-VL-8B-Instruct-8bit"


class TestMakeDecision:
    """Tests for decision-making logic."""

    def test_no_frames_returns_keep(self):
        """Test that empty frame list returns KEEP with LOW confidence."""
        decision, confidence, in_point, out_point = _make_decision([], 10.0, None)

        assert decision == ClipDecision.KEEP
        assert confidence == ConfidenceLevel.LOW
        assert in_point is None
        assert out_point is None

    def test_all_good_frames_returns_keep(self):
        """Test that all good frames returns KEEP."""
        frames = [
            FrameAnalysis(
                frame_index=i,
                timestamp_seconds=float(i),
                is_junk=False,
                blur_score=200.0,
            )
            for i in range(8)
        ]

        decision, confidence, in_point, out_point = _make_decision(frames, 8.0, -20.0)

        assert decision == ClipDecision.KEEP
        assert in_point is None
        assert out_point is None

    def test_majority_junk_returns_reject(self):
        """Test that >50% junk frames returns REJECT."""
        frames = [
            FrameAnalysis(
                frame_index=i,
                timestamp_seconds=float(i),
                is_junk=True,
                blur_score=50.0,
            )
            for i in range(6)
        ]
        # Add 2 good frames
        frames.extend(
            [
                FrameAnalysis(
                    frame_index=6,
                    timestamp_seconds=6.0,
                    is_junk=False,
                    blur_score=200.0,
                ),
                FrameAnalysis(
                    frame_index=7,
                    timestamp_seconds=7.0,
                    is_junk=False,
                    blur_score=200.0,
                ),
            ]
        )

        decision, confidence, in_point, out_point = _make_decision(frames, 8.0, -20.0)

        assert decision == ClipDecision.REJECT

    def test_start_junk_suggests_in_point(self):
        """Test that junk at start suggests trimming."""
        frames = [
            # First 4 frames are junk (at 0-3s)
            FrameAnalysis(frame_index=0, timestamp_seconds=0.0, is_junk=True),
            FrameAnalysis(frame_index=1, timestamp_seconds=1.0, is_junk=True),
            FrameAnalysis(frame_index=2, timestamp_seconds=2.0, is_junk=True),
            FrameAnalysis(frame_index=3, timestamp_seconds=3.0, is_junk=True),
            # Rest are good
            FrameAnalysis(frame_index=4, timestamp_seconds=5.0, is_junk=False),
            FrameAnalysis(frame_index=5, timestamp_seconds=6.0, is_junk=False),
            FrameAnalysis(frame_index=6, timestamp_seconds=7.0, is_junk=False),
            FrameAnalysis(frame_index=7, timestamp_seconds=8.0, is_junk=False),
        ]

        decision, confidence, in_point, out_point = _make_decision(frames, 10.0, -20.0)

        assert decision == ClipDecision.REVIEW
        assert in_point == 3.0
        assert out_point is None

    def test_very_low_audio_returns_review(self):
        """Test that very low audio triggers review."""
        frames = [
            FrameAnalysis(
                frame_index=0,
                timestamp_seconds=0.0,
                is_junk=False,
                blur_score=200.0,
            )
        ]

        decision, confidence, in_point, out_point = _make_decision(frames, 5.0, -55.0)

        assert decision == ClipDecision.REVIEW
        assert confidence == ConfidenceLevel.LOW


class TestJunkReason:
    """Tests for JunkReason enum."""

    def test_junk_reason_values(self):
        """Test junk reason values."""
        assert JunkReason.BLUR.value == "blur"
        assert JunkReason.DARKNESS.value == "darkness"
        assert JunkReason.LENS_CAP.value == "lens_cap"
        assert JunkReason.GROUND.value == "pointing_at_ground"
        assert JunkReason.ACCIDENTAL.value == "accidental_trigger"
        assert JunkReason.LOW_AUDIO.value == "low_audio"


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_confidence_level_values(self):
        """Test confidence level values."""
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"


class TestClipDecision:
    """Tests for ClipDecision enum."""

    def test_clip_decision_values(self):
        """Test clip decision values."""
        assert ClipDecision.KEEP.value == "keep"
        assert ClipDecision.REJECT.value == "reject"
        assert ClipDecision.REVIEW.value == "review"

"""Tests for the timestamp fixer tool."""

from datetime import datetime, timedelta
from pathlib import Path
import tempfile

import pytest

from tvas.tools.timestamp_fixer import TimeShiftEngine, ClipInfo


class TestTimeShiftEngine:
    """Test cases for TimeShiftEngine."""
    
    def test_parse_time_delta_positive(self):
        """Test parsing positive time delta."""
        engine = TimeShiftEngine()
        
        delta = engine.parse_time_delta("+01:30:00")
        assert delta == timedelta(hours=1, minutes=30)
        
        delta = engine.parse_time_delta("+00:05:30")
        assert delta == timedelta(minutes=5, seconds=30)
        
        delta = engine.parse_time_delta("02:00:00")
        assert delta == timedelta(hours=2)
    
    def test_parse_time_delta_negative(self):
        """Test parsing negative time delta."""
        engine = TimeShiftEngine()
        
        delta = engine.parse_time_delta("-01:00:00")
        assert delta == timedelta(hours=-1)
        
        delta = engine.parse_time_delta("-00:30:00")
        assert delta == timedelta(minutes=-30)
    
    def test_parse_time_delta_short_format(self):
        """Test parsing HH:MM format."""
        engine = TimeShiftEngine()
        
        delta = engine.parse_time_delta("+01:30")
        assert delta == timedelta(hours=1, minutes=30)
        
        delta = engine.parse_time_delta("-00:15")
        assert delta == timedelta(minutes=-15)
    
    def test_parse_time_delta_invalid(self):
        """Test parsing invalid time formats."""
        engine = TimeShiftEngine()
        
        assert engine.parse_time_delta("") is None
        assert engine.parse_time_delta("invalid") is None
        assert engine.parse_time_delta("1:2:3:4") is None
    
    def test_format_time_delta(self):
        """Test formatting time delta."""
        engine = TimeShiftEngine()
        
        assert engine.format_time_delta(timedelta(hours=1, minutes=30)) == "+01:30:00"
        assert engine.format_time_delta(timedelta(hours=-2)) == "-02:00:00"
        assert engine.format_time_delta(timedelta(minutes=5, seconds=30)) == "+00:05:30"
    
    def test_calculate_sync_offset(self):
        """Test calculating sync offset between clips."""
        engine = TimeShiftEngine()
        
        clip_a = ClipInfo(
            path=Path("/test/a.mp4"),
            camera="CameraA",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            duration_seconds=60.0,
            file_size=1000,
        )
        
        clip_b = ClipInfo(
            path=Path("/test/b.mp4"),
            camera="CameraB",
            created_at=datetime(2024, 1, 1, 11, 0, 0),
            duration_seconds=60.0,
            file_size=1000,
        )
        
        # clip_a is 1 hour ahead of clip_b
        offset = engine.calculate_sync_offset(clip_a, clip_b)
        assert offset == timedelta(hours=1)
        
        # Reverse - clip_b is 1 hour behind clip_a
        offset = engine.calculate_sync_offset(clip_b, clip_a)
        assert offset == timedelta(hours=-1)
    
    def test_clip_info_end_time(self):
        """Test ClipInfo end_time property."""
        clip = ClipInfo(
            path=Path("/test/clip.mp4"),
            camera="TestCamera",
            created_at=datetime(2024, 1, 1, 10, 0, 0),
            duration_seconds=120.0,
            file_size=1000,
        )
        
        expected_end = datetime(2024, 1, 1, 10, 2, 0)
        assert clip.end_time == expected_end
    
    def test_detect_anomalies(self):
        """Test anomaly detection within camera clips."""
        engine = TimeShiftEngine()
        
        # Create clips with a large gap
        engine.clips = [
            ClipInfo(
                path=Path("/test/clip1.mp4"),
                camera="CameraA",
                created_at=datetime(2024, 1, 1, 10, 0, 0),
                duration_seconds=60.0,
                file_size=1000,
            ),
            ClipInfo(
                path=Path("/test/clip2.mp4"),
                camera="CameraA",
                created_at=datetime(2024, 1, 1, 10, 2, 0),  # 1 min after clip1 ends
                duration_seconds=60.0,
                file_size=1000,
            ),
            ClipInfo(
                path=Path("/test/clip3.mp4"),
                camera="CameraA",
                created_at=datetime(2024, 1, 1, 20, 0, 0),  # 10 hours later!
                duration_seconds=60.0,
                file_size=1000,
            ),
        ]
        
        engine.cameras = {"CameraA": engine.clips}
        
        anomalies = engine.detect_anomalies(threshold_hours=6)
        
        assert len(anomalies) == 1
        assert anomalies[0].path == Path("/test/clip3.mp4")
        assert anomalies[0].is_anomaly is True
        assert anomalies[0].anomaly_gap_hours > 9  # ~10 hours gap
    
    def test_get_clips_in_range(self):
        """Test filtering clips by time range."""
        engine = TimeShiftEngine()
        
        engine.clips = [
            ClipInfo(
                path=Path("/test/clip1.mp4"),
                camera="CameraA",
                created_at=datetime(2024, 1, 1, 10, 0, 0),
                duration_seconds=60.0,
                file_size=1000,
            ),
            ClipInfo(
                path=Path("/test/clip2.mp4"),
                camera="CameraA",
                created_at=datetime(2024, 1, 1, 12, 0, 0),
                duration_seconds=60.0,
                file_size=1000,
            ),
            ClipInfo(
                path=Path("/test/clip3.mp4"),
                camera="CameraB",
                created_at=datetime(2024, 1, 1, 11, 0, 0),
                duration_seconds=60.0,
                file_size=1000,
            ),
        ]
        
        # Get all clips in range
        result = engine.get_clips_in_range(
            datetime(2024, 1, 1, 9, 0, 0),
            datetime(2024, 1, 1, 13, 0, 0)
        )
        assert len(result) == 3
        
        # Get clips from specific camera
        result = engine.get_clips_in_range(
            datetime(2024, 1, 1, 9, 0, 0),
            datetime(2024, 1, 1, 13, 0, 0),
            cameras=["CameraA"]
        )
        assert len(result) == 2
        
        # Get clips in narrow range
        result = engine.get_clips_in_range(
            datetime(2024, 1, 1, 10, 30, 0),
            datetime(2024, 1, 1, 11, 30, 0)
        )
        assert len(result) == 1
        assert result[0].camera == "CameraB"


class TestClipInfo:
    """Test cases for ClipInfo dataclass."""
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        clip = ClipInfo(
            path=Path("/test/clip.mp4"),
            camera="TestCamera",
            created_at=datetime(2024, 1, 1, 10, 0, 0),
            duration_seconds=120.0,
            file_size=1024000,
            is_anomaly=True,
            anomaly_gap_hours=8.5,
        )
        
        d = clip.to_dict()
        
        assert d["path"] == "/test/clip.mp4"
        assert d["camera"] == "TestCamera"
        assert d["created_at"] == "2024-01-01T10:00:00"
        assert d["duration_seconds"] == 120.0
        assert d["file_size"] == 1024000
        assert d["is_anomaly"] is True
        assert d["anomaly_gap_hours"] == 8.5

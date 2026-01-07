"""Tests for the trim module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tvas.trim import detect_trims_batch, generate_trim_proxy, detect_trim_for_file

class TestTrimDetection:
    @pytest.fixture
    def mock_vlm_client(self):
        with patch("tvas.trim.VLMClient") as mock:
            client = mock.return_value
            yield client

    @pytest.fixture
    def mock_ffmpeg(self):
        with patch("subprocess.run") as mock:
            mock.return_value.returncode = 0
            yield mock
            
    @pytest.fixture
    def mock_duration(self):
        with patch("tvas.trim.get_video_duration", return_value=60.0) as mock:
            yield mock
            
    @pytest.fixture
    def mock_check_ffmpeg(self):
        with patch("tvas.trim.check_ffmpeg_available", return_value=True) as mock:
            yield mock

    def test_generate_trim_proxy(self, tmp_path, mock_ffmpeg, mock_duration, mock_check_ffmpeg):
        video = tmp_path / "video.mp4"
        video.touch()
        
        # Should generate a temp file (NamedTemporaryFile creates it on disk)
        proxy = generate_trim_proxy(video)
        assert proxy is not None
        assert proxy.exists()
        
        # Verify FFmpeg call
        # args[0] is the command list
        cmd = mock_ffmpeg.call_args[0][0]
        # Check for filter complex string in arguments
        assert any("trim=0:5.0" in arg for arg in cmd)
        
        if proxy != video and proxy.exists():
            proxy.unlink()

    def test_detect_trim_parsing(self, tmp_path, mock_vlm_client, mock_ffmpeg, mock_duration, mock_check_ffmpeg):
        # Setup
        video = tmp_path / "video.mp4"
        video.touch()
        json_path = tmp_path / "video.json"
        json_path.write_text(json.dumps({
            "source_path": str(video),
            "metadata": {"duration_seconds": 60.0}
        }))
        
        # Mock VLM response
        # Scenario: Start trim at 2.0 (in first 5s), End trim at 7.0 (relative to 10s stitched)
        # Stitched: 0-5s (original 0-5), 5-10s (original 55-60)
        # Rel 2.0 -> Orig 2.0
        # Rel 7.0 -> Orig 55 + (7-5) = 57.0
        mock_response = MagicMock()
        mock_response.text = '{"trim_needed": true, "start_sec": 2.0, "end_sec": 7.0, "reason": "shake"}'
        mock_vlm_client.generate_from_video.return_value = mock_response
        
        # Run
        detect_trim_for_file(json_path, mock_vlm_client)
        
        # Verify JSON update
        data = json.loads(json_path.read_text())
        
        # Check nested trim object
        assert "trim" in data
        trim_data = data["trim"]
        assert trim_data["trim_needed"] is True
        assert trim_data["suggested_in_point"] == 2.0
        assert trim_data["suggested_out_point"] == 57.0
        assert trim_data["reason"] == "shake"

    def test_skip_existing_trim(self, tmp_path, mock_vlm_client):
        # Setup
        json_path = tmp_path / "existing.json"
        json_path.write_text(json.dumps({
            "source_path": "video.mp4",
            "trim": {"trim_needed": False}
        }))
        
        # Run
        detect_trim_for_file(json_path, mock_vlm_client)
        
        # Verify VLM was NOT called
        mock_vlm_client.generate_from_video.assert_not_called()

    def test_skip_removed_clips(self, tmp_path, mock_vlm_client):
        json_path = tmp_path / "removed.json"
        json_path.write_text(json.dumps({
            "source_path": "video.mp4",
            "beat": {"classification": "REMOVE"}
        }))
        
        detect_trim_for_file(json_path, mock_vlm_client)
        
        mock_vlm_client.generate_from_video.assert_not_called()
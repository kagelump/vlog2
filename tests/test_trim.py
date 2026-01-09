"""Tests for the trim module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tvas.trim import detect_trims_batch, detect_trim_for_file

class TestTrimDetection:
    @pytest.fixture
    def mock_vlm_client(self):
        with patch("tvas.trim.VLMClient") as mock:
            client = mock.return_value
            yield client

    def test_detect_trim_parsing(self, tmp_path, mock_vlm_client):
        # Setup
        video = tmp_path / "video.mp4"
        video.touch()
        json_path = tmp_path / "video.json"
        json_path.write_text(json.dumps({
            "source_path": str(video),
            "metadata": {"duration_seconds": 60.0}
        }))
        
        # Mock VLM response with new format
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "best_moment": {
                "start_sec": 5.0,
                "end_sec": 9.0,
                "score": 9
            },
            "technical_trim": {
                "trim_needed": True,
                "start_sec": 1.0,
                "end_sec": 58.0,
                "reason": "shake"
            },
            "action_peaks": [10.0, 20.0],
            "dead_zones": [{"start_sec": 30.0, "end_sec": 35.0}]
        })
        mock_vlm_client.generate_from_video.return_value = mock_response
        
        # Run
        detect_trim_for_file(json_path, mock_vlm_client)
        
        # Verify JSON update
        data = json.loads(json_path.read_text())
        
        # Check nested trim object
        assert "trim" in data
        trim_data = data["trim"]
        assert trim_data["trim_needed"] is True
        assert trim_data["suggested_in_point"] == 1.0
        assert trim_data["suggested_out_point"] == 58.0
        assert trim_data["best_moment"]["score"] == 9
        assert trim_data["action_peaks"] == [10.0, 20.0]
        assert len(trim_data["dead_zones"]) == 1

    def test_skip_existing_trim(self, tmp_path, mock_vlm_client):
        # Setup
        json_path = tmp_path / "existing.json"
        json_path.write_text(json.dumps({
            "source_path": "video.mp4",
            "trim": {
                "best_moment": {"start_sec": 1.0, "end_sec": 4.0, "score": 8}
            }
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

    def test_skip_hero_clips(self, tmp_path, mock_vlm_client):
        json_path = tmp_path / "hero.json"
        json_path.write_text(json.dumps({
            "source_path": "video.mp4",
            "beat": {"classification": "HERO"}
        }))
        
        detect_trim_for_file(json_path, mock_vlm_client)
        
        mock_vlm_client.generate_from_video.assert_not_called()

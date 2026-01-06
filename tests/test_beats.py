"""Tests for the beats module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tvas.beats import align_beats

class TestBeatsAlignment:
    @pytest.fixture
    def mock_vlm_client(self):
        with patch("tvas.beats.VLMClient") as mock:
            client_instance = mock.return_value
            # Mock generate response
            mock_response = MagicMock()
            mock_response.text = '{"beat_id": "Beat 1", "beat_title": "Test Beat", "classification": "HIGHLIGHT", "reasoning": "Fits well"}'
            client_instance.generate.return_value = mock_response
            yield client_instance

    @pytest.fixture
    def mock_aggregate(self):
        with patch("tvas.beats.aggregate_analysis_json") as mock:
            yield mock

    def test_align_beats(self, tmp_path, mock_vlm_client, mock_aggregate):
        # Setup
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        
        outline_path = tmp_path / "outline.md"
        outline_path.write_text("# Outline\n\nBeat 1: Test Beat\nGoal: Test")
        
        # Create dummy clips and json
        clip1 = project_dir / "clip1.mp4"
        clip1.touch()
        json1 = project_dir / "clip1.json"
        json1.write_text(json.dumps({
            "clip_name": "clip1",
            "metadata": {"created_timestamp": "2023-01-01"}
        }))
        
        # Create thumbnail
        thumb1 = project_dir / "clip1.jpg"
        thumb1.touch()
        
        # Run alignment
        align_beats(
            project_dir=project_dir,
            outline_path=outline_path,
        )
        
        # Verify VLM was called
        assert mock_vlm_client.generate.call_count == 1
        
        # Verify JSON was updated
        data = json.loads(json1.read_text())
        assert "beat" in data
        assert data["beat"]["beat_id"] == "Beat 1"
        
        # Verify aggregation was called
        mock_aggregate.assert_called_once_with(project_dir)
        
    def test_thumbnail_generation_called(self, tmp_path, mock_vlm_client, mock_aggregate):
        # Setup
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        
        outline_path = tmp_path / "outline.md"
        outline_path.write_text("# Outline")
        
        # Create json with thumbnail_timestamp_sec but NO thumbnail jpg
        json1 = project_dir / "clip1.json"
        json1.write_text(json.dumps({
            "clip_name": "clip1",
            "thumbnail_timestamp_sec": 5.0,
            "source_path": str(project_dir / "clip1.mp4")
        }))
        
        # Create dummy video file
        video1 = project_dir / "clip1.mp4"
        video1.touch()
        
        with patch("tvas.beats.extract_frame") as mock_extract:
            align_beats(project_dir, outline_path)
            # Verify extract_frame was called
            mock_extract.assert_called_once()
            args, kwargs = mock_extract.call_args
            assert args[0] == video1
            assert args[1] == 5.0
            assert args[2] == project_dir / "clip1.jpg"
            assert kwargs.get("max_dimension") == 512

    def test_skip_existing_beat(self, tmp_path, mock_vlm_client, mock_aggregate):
        # Setup
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        
        outline_path = tmp_path / "outline.md"
        outline_path.write_text("# Outline")
        
        json1 = project_dir / "clip1.json"
        json1.write_text(json.dumps({
            "clip_name": "clip1",
            "beat": {"beat_id": "Existing"}
        }))
        
        align_beats(project_dir, outline_path)
        
        # Verify VLM was NOT called
        mock_vlm_client.generate.assert_not_called()
        
        # Verify aggregation WAS called (even if skipped, we re-aggregate)
        mock_aggregate.assert_called_once_with(project_dir)

    def test_sorting_by_timestamp(self, tmp_path, mock_vlm_client, mock_aggregate):
        # Setup
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        
        outline_path = tmp_path / "outline.md"
        outline_path.write_text("# Outline")
        
        # Clip B: later timestamp, "earlier" name
        json_b = project_dir / "a_clip.json"
        json_b.write_text(json.dumps({
            "clip_name": "B",
            "metadata": {"created_timestamp": "2023-01-01 12:00:00"}
        }))
        
        # Clip A: earlier timestamp, "later" name
        json_a = project_dir / "z_clip.json"
        json_a.write_text(json.dumps({
            "clip_name": "A",
            "metadata": {"created_timestamp": "2023-01-01 10:00:00"}
        }))
        
        # We want to verify that Clip A (z_clip.json) is processed BEFORE Clip B (a_clip.json)
        # despite the alphabetical order of filenames.
        
        with patch.object(mock_vlm_client, 'generate', side_effect=mock_vlm_client.generate) as mock_gen:
            align_beats(project_dir, outline_path)
            
            # Check order of calls
            # Call 1 should be for Clip A (10:00:00)
            # Call 2 should be for Clip B (12:00:00)
            calls = mock_gen.call_args_list
            assert "2023-01-01 10:00:00" in str(calls[0])
            assert "2023-01-01 12:00:00" in str(calls[1])

    def test_missing_outline(self, tmp_path, caplog):
        align_beats(tmp_path, tmp_path / "nonexistent.md")
        assert "Outline file not found" in caplog.text
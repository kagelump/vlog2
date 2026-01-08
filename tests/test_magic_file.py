"""Tests for the magic file creation."""

from pathlib import Path
from unittest.mock import patch
import pytest
from tvas.analysis import aggregate_analysis_json, ClipAnalysis

class TestMagicFile:
    @pytest.fixture
    def mock_home(self, tmp_path):
        """Mock Path.home() to return a temporary directory."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            yield tmp_path

    def test_aggregate_analysis_json_creates_magic_file(self, tmp_path, mock_home):
        # Create a dummy json file to aggregate
        json_file = tmp_path / "clip.json"
        json_file.write_text('{"metadata": {"created_timestamp": "2023"}}')
        
        aggregate_analysis_json(tmp_path)
        
        magic_file = mock_home / ".tvas_current_analysis"
        assert magic_file.exists()
        expected_path = tmp_path / "analysis.json"
        assert magic_file.read_text() == str(expected_path.resolve())

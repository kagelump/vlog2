import csv
from pathlib import Path
import pytest
from tvas.analysis import aggregate_analysis_csv

class TestCSVExport:
    def test_aggregate_analysis_csv(self, tmp_path):
        """Test aggregation from raw dictionary data (simulating JSON)."""
        data = [{
            "source_path": "video.mp4",
            "metadata": {
                "created_timestamp": "2023-01-01"
            },
            # Map 'trim' -> 'needs_trim' via json_key_map
            "trim": True,
            # Map 'start_sec' -> 'suggested_in_point'
            "start_sec": 1.5,
            "subject_keywords": ["apple", "banana"]
        }]
        
        csv_path = aggregate_analysis_csv(tmp_path, data)
        
        assert csv_path.exists()
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 1
        row = rows[0]
        
        assert row["Source Path"] == "video.mp4"
        assert row["Needs Trim"] == "Yes"
        assert row["Suggested In Point"] == "1.50"
        assert row["Subject Keywords"] == "apple, banana"

    def test_aggregate_analysis_csv_nested_beats(self, tmp_path):
        """Test aggregation with nested beat fields."""
        data = [{
            "source_path": "video.mp4",
            "metadata": {},
            "beat": {
                "beat_id": "Beat 1",
                "beat_title": "The Intro",
                "classification": "HERO",
                "reasoning": "Fits well"
            }
        }]
        
        csv_path = aggregate_analysis_csv(tmp_path, data)
        
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 1
        row = rows[0]
        assert row["Beat Id"] == "Beat 1"
        assert row["Beat Title"] == "The Intro"
        assert row["Beat Classification"] == "HERO"
        assert row["Beat Reasoning"] == "Fits well"

    def test_empty_list(self, tmp_path):
        """Test export with empty list."""
        csv_path = aggregate_analysis_csv(tmp_path, [])
        
        assert csv_path.exists()
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Should just have headers
        assert len(lines) == 1
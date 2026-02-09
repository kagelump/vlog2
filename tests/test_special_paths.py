"""Tests for paths with special characters (spaces, parens, unicode) and case-insensitive extensions.

Verifies that video file scanning works correctly regardless of file extension casing
and that paths containing spaces, parentheses, and other special characters are handled.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tvas.main import TVASApp
from tvas.trim import detect_trim_for_file
from tvas.analysis import aggregate_analysis_json


# ---------- Case-insensitive extension scanning ----------


class TestCaseInsensitiveExtensions:
    """Ensure video file discovery works with any extension casing."""

    VIDEO_EXTENSIONS_CASES = [
        ".mp4", ".MP4", ".Mp4",
        ".mov", ".MOV", ".Mov",
        ".mts", ".MTS",
        ".mxf", ".MXF",
        ".insv", ".INSV",
        ".insp", ".INSP",
    ]

    def _populate_dir(self, directory: Path, filenames: list[str]):
        """Helper to create empty files in a directory."""
        directory.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            (directory / name).touch()

    # -- process_directory (main.py) --

    def test_process_directory_finds_uppercase_mp4(self, tmp_path):
        """process_directory should find .MP4 files."""
        

        self._populate_dir(tmp_path, ["clip1.MP4", "clip2.mp4", "clip3.Mp4"])

        proc = TVASApp(use_vlm=False)
        # We only care about file discovery, so patch the actual analysis
        with patch("tvas.main.analyze_clips_batch", return_value=[]) as mock_analyze:
            result = proc.process_directory(tmp_path)

        # All 3 files should be discovered
        assert result["files_processed"] == 3

    def test_process_directory_finds_mixed_extensions(self, tmp_path):
        """process_directory should find all video types regardless of casing."""
        

        self._populate_dir(tmp_path, [
            "a.MOV", "b.mov", "c.MTS", "d.insv", "e.INSP", "f.MXF",
        ])

        proc = TVASApp(use_vlm=False)
        with patch("tvas.main.analyze_clips_batch", return_value=[]):
            result = proc.process_directory(tmp_path)

        assert result["files_processed"] == 6

    def test_process_directory_ignores_hidden_files(self, tmp_path):
        """Hidden files should be skipped even with valid extensions."""
        

        self._populate_dir(tmp_path, [".hidden.mp4", "visible.MP4"])

        proc = TVASApp(use_vlm=False)
        with patch("tvas.main.analyze_clips_batch", return_value=[]):
            result = proc.process_directory(tmp_path)

        assert result["files_processed"] == 1

    # -- process_archival (main.py) --

    def test_process_archival_finds_uppercase_in_camera_dirs(self, tmp_path):
        """process_archival should find .MP4 in camera subdirectories."""
        

        cam_dir = tmp_path / "SonyA7C"
        self._populate_dir(cam_dir, ["C0001.MP4", "C0002.mp4", "C0003.MOV"])

        proc = TVASApp(use_vlm=False)
        with patch.object(proc, "_process_pipeline", return_value={"success": True}) as mock_pipe:
            result = proc.process_from_archival(tmp_path)

        # _process_pipeline should receive all 3 files
        source_files = mock_pipe.call_args[0][0]
        assert len(source_files) == 3


    # -- trim fallback resolution --

    def test_trim_fallback_finds_uppercase_mp4(self, tmp_path):
        """Trim detection should fall back to .MP4 when .mp4 doesn't exist."""
        video = tmp_path / "clip.MP4"
        video.touch()

        json_path = tmp_path / "clip.json"
        json_path.write_text(json.dumps({
            "metadata": {"duration_seconds": 30.0},
        }))

        with patch("tvas.trim.VLMClient") as mock_cls:
            client = mock_cls.return_value
            mock_resp = MagicMock()
            mock_resp.text = json.dumps({
                "best_moment": {"start_sec": 1.0, "end_sec": 5.0, "score": 7},
                "technical_trim": {"trim_needed": False},
                "action_peaks": [],
                "dead_zones": [],
            })
            client.generate_from_video.return_value = mock_resp

            result = detect_trim_for_file(json_path, client)

        assert result is True  # Should have found and processed the .MP4

    def test_trim_fallback_prefers_lowercase_mp4(self, tmp_path):
        """When both .mp4 and .MP4 exist, lowercase should be used first."""
        lower_video = tmp_path / "clip.mp4"
        lower_video.touch()
        upper_video = tmp_path / "clip.MP4"
        upper_video.touch()

        json_path = tmp_path / "clip.json"
        json_path.write_text(json.dumps({
            "metadata": {"duration_seconds": 30.0},
        }))

        with patch("tvas.trim.VLMClient") as mock_cls:
            client = mock_cls.return_value
            mock_resp = MagicMock()
            mock_resp.text = json.dumps({
                "best_moment": {"start_sec": 1.0, "end_sec": 5.0, "score": 7},
                "technical_trim": {"trim_needed": False},
                "action_peaks": [],
                "dead_zones": [],
            })
            client.generate_from_video.return_value = mock_resp

            detect_trim_for_file(json_path, client)

        call_args = client.generate_from_video.call_args
        video_used = call_args[1].get("video_path") or call_args[0][0]
        assert Path(str(video_used)).suffix == ".mp4"


# ---------- Paths with special characters ----------


class TestSpecialCharacterPaths:
    """Ensure paths with spaces, parentheses, and unicode work correctly."""

    SPECIAL_DIR_NAMES = [
        "My Project",
        "Tokyo (Day 1)",
        "2024 Trip (Summer)",
        "Project - Test & More",
        "日本旅行",
    ]

    def test_process_directory_with_spaces_and_parens(self, tmp_path):
        """process_directory works with spaces and parentheses in path."""
        

        project_dir = tmp_path / "My Project (2024)" / "Camera A"
        project_dir.mkdir(parents=True)
        # Put files directly in the special-char dir (not in Camera A subdir,
        # because process_directory scans the given dir directly)
        target = tmp_path / "My Project (2024)"
        (target / "clip.MP4").touch()

        proc = TVASApp(use_vlm=False)
        with patch("tvas.main.analyze_clips_batch", return_value=[]):
            result = proc.process_directory(target)

        assert result["files_processed"] == 1

    def test_process_archival_with_special_chars(self, tmp_path):
        """process_archival works when camera folders have special chars."""
        

        archival = tmp_path / "Bla blah" / "blah (blah)"
        cam_dir = archival / "Sony A7C (Cam 1)"
        cam_dir.mkdir(parents=True)
        (cam_dir / "C0001.MP4").touch()
        (cam_dir / "C0002.mov").touch()

        proc = TVASApp(use_vlm=False)
        with patch.object(proc, "_process_pipeline", return_value={"success": True}) as mock_pipe:
            result = proc.process_from_archival(archival)

        source_files = mock_pipe.call_args[0][0]
        assert len(source_files) == 2

    def test_trim_with_special_path(self, tmp_path):
        """Trim detection works when video has spaces/parens in path."""
        special_dir = tmp_path / "My Trip (Summer 2024)"
        special_dir.mkdir(parents=True)

        video = special_dir / "clip with spaces.MP4"
        video.touch()

        json_path = special_dir / "clip with spaces.json"
        json_path.write_text(json.dumps({
            "source_path": str(video),
            "metadata": {"duration_seconds": 30.0},
        }))

        with patch("tvas.trim.VLMClient") as mock_cls:
            client = mock_cls.return_value
            mock_resp = MagicMock()
            mock_resp.text = json.dumps({
                "best_moment": {"start_sec": 1.0, "end_sec": 5.0, "score": 7},
                "technical_trim": {"trim_needed": False},
                "action_peaks": [],
                "dead_zones": [],
            })
            client.generate_from_video.return_value = mock_resp

            result = detect_trim_for_file(json_path, client)

        assert result is True

    def test_aggregate_analysis_with_special_path(self, tmp_path):
        """aggregate_analysis_json works with special chars in dir path."""
        project_dir = tmp_path / "Trip (Day 1)"
        project_dir.mkdir(parents=True)

        # Create a clip JSON with uppercase extension source
        clip_json = project_dir / "clip001.json"
        clip_data = {
            "metadata": {"created_timestamp": "2024-01-01T12:00:00"},
            "source_path": str(project_dir / "clip001.MP4"),
        }
        clip_json.write_text(json.dumps(clip_data))

        # Create the video file so the source_path is valid
        (project_dir / "clip001.MP4").touch()

        result = aggregate_analysis_json(project_dir)
        assert result is not None  # Should not crash


# ---------- sys.argv clearing for Toga/Cocoa ----------


class TestSysArgvClearing:
    """Verify that ui.main() clears sys.argv to prevent Cocoa document errors."""

    def test_main_clears_argv(self):
        """ui.main() should strip CLI args so Cocoa doesn't try to open them."""
        original_argv = sys.argv[:]
        try:
            sys.argv = [
                "tvas",
                "--project",
                "/Volumes/Bla blah/blah (blah)/blah",
                "--model",
                "test",
            ]

            # Mock TvasStatusApp so we don't actually create a GUI
            with patch("tvas.ui.TvasStatusApp") as mock_app_cls:
                mock_app_cls.return_value = MagicMock()
                from tvas.ui import main as ui_main
                ui_main(project_path=Path("/Volumes/Bla blah/blah (blah)/blah"))

            # sys.argv should now only contain the program name
            assert len(sys.argv) == 1
            assert sys.argv[0] == "tvas"
        finally:
            sys.argv = original_argv

    def test_main_preserves_program_name(self):
        """sys.argv[0] should be preserved after clearing."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["/usr/local/bin/tvas", "--verbose", "--project", "/some/path"]

            with patch("tvas.ui.TvasStatusApp") as mock_app_cls:
                mock_app_cls.return_value = MagicMock()
                from tvas.ui import main as ui_main
                ui_main()

            assert sys.argv == ["/usr/local/bin/tvas"]
        finally:
            sys.argv = original_argv

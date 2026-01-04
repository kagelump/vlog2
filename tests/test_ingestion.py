"""Tests for the ingestion module."""

import hashlib
import tempfile
from pathlib import Path

import pytest

from tvas.ingestion import (
    CameraType,
    calculate_sha256,
    copy_file_with_progress,
    create_destination_structure,
    detect_camera_type,
    get_video_files,
    verify_copy,
)


class TestCameraDetection:
    """Tests for camera type detection."""

    def test_detect_unknown_empty_volume(self, tmp_path: Path):
        """Test detection returns UNKNOWN for empty volume."""
        result = detect_camera_type(tmp_path)
        assert result == CameraType.UNKNOWN

    def test_detect_sony_a7c(self, tmp_path: Path):
        """Test detection of Sony A7C camera."""
        # Create Sony directory structure
        sony_path = tmp_path / "PRIVATE" / "M4ROOT" / "CLIP"
        sony_path.mkdir(parents=True)
        (sony_path / "test.MP4").touch()

        result = detect_camera_type(tmp_path)
        assert result == CameraType.SONY_A7C

    def test_detect_dji_pocket3(self, tmp_path: Path):
        """Test detection of DJI Pocket 3 camera."""
        # Create DJI directory structure
        dji_path = tmp_path / "DCIM" / "100MEDIA"
        dji_path.mkdir(parents=True)
        (dji_path / "DJI_0001.MP4").touch()

        result = detect_camera_type(tmp_path)
        assert result == CameraType.DJI_POCKET3

    def test_detect_iphone(self, tmp_path: Path):
        """Test detection of iPhone."""
        # Create iPhone directory structure
        iphone_path = tmp_path / "DCIM" / "100APPLE"
        iphone_path.mkdir(parents=True)
        (iphone_path / "IMG_0001.MOV").touch()

        result = detect_camera_type(tmp_path)
        assert result == CameraType.IPHONE_11PRO

    def test_detect_insta360(self, tmp_path: Path):
        """Test detection of Insta360 camera."""
        # Create Insta360 directory structure
        insta_path = tmp_path / "DCIM" / "Camera01"
        insta_path.mkdir(parents=True)
        (insta_path / "VID_001.insv").touch()

        result = detect_camera_type(tmp_path)
        assert result == CameraType.INSTA360


class TestChecksumCalculation:
    """Tests for SHA256 checksum calculation."""

    def test_calculate_sha256(self, tmp_path: Path):
        """Test SHA256 calculation."""
        test_file = tmp_path / "test.txt"
        content = b"Hello, World!"
        test_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        result = calculate_sha256(test_file)

        assert result == expected

    def test_calculate_sha256_large_file(self, tmp_path: Path):
        """Test SHA256 calculation for larger file."""
        test_file = tmp_path / "large.bin"
        content = b"x" * (1024 * 1024)  # 1MB
        test_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        result = calculate_sha256(test_file)

        assert result == expected


class TestFileCopy:
    """Tests for file copying with verification."""

    def test_copy_file_with_progress(self, tmp_path: Path):
        """Test copying a file with progress callback."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest" / "dest.txt"
        content = b"Test content for copying"
        source.write_bytes(content)

        progress_calls = []

        def progress_cb(copied, total):
            progress_calls.append((copied, total))

        result = copy_file_with_progress(source, dest, progress_cb)

        assert result == dest
        assert dest.exists()
        assert dest.read_bytes() == content
        assert len(progress_calls) > 0

    def test_verify_copy_success(self, tmp_path: Path):
        """Test successful copy verification."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        content = b"Test content"
        source.write_bytes(content)
        dest.write_bytes(content)

        assert verify_copy(source, dest) is True

    def test_verify_copy_failure(self, tmp_path: Path):
        """Test copy verification failure."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_bytes(b"Original content")
        dest.write_bytes(b"Different content")

        assert verify_copy(source, dest) is False

    def test_verify_copy_missing_dest(self, tmp_path: Path):
        """Test copy verification with missing destination."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "nonexistent.txt"
        source.write_bytes(b"Content")

        assert verify_copy(source, dest) is False


class TestGetVideoFiles:
    """Tests for video file discovery."""

    def test_get_video_files_sony(self, tmp_path: Path):
        """Test getting video files from Sony camera."""
        sony_path = tmp_path / "PRIVATE" / "M4ROOT" / "CLIP"
        sony_path.mkdir(parents=True)
        (sony_path / "test1.MP4").write_bytes(b"fake video")
        (sony_path / "test2.mp4").write_bytes(b"fake video")
        (sony_path / "other.txt").write_bytes(b"not video")

        files = get_video_files(tmp_path, CameraType.SONY_A7C)

        assert len(files) == 2
        assert all(f.camera_type == CameraType.SONY_A7C for f in files)

    def test_get_video_files_empty_volume(self, tmp_path: Path):
        """Test getting files from empty volume."""
        files = get_video_files(tmp_path, CameraType.UNKNOWN)
        assert len(files) == 0

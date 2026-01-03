"""Tests for the TPRS (Travel Photo Rating System) module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree as ET

import pytest

from tprs.tprs import (
    PhotoAnalysis,
    find_jpeg_photos,
    generate_xmp_sidecar,
)
from shared import DEFAULT_VLM_MODEL


class TestFindJpegPhotos:
    """Tests for finding JPEG photos."""

    def test_find_jpeg_photos_empty_directory(self):
        """Test finding photos in an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            photos = find_jpeg_photos(Path(temp_dir))
            assert photos == []

    def test_find_jpeg_photos_with_jpegs(self):
        """Test finding JPEG photos in a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some test JPEG files
            (temp_path / "photo1.jpg").touch()
            (temp_path / "photo2.jpeg").touch()
            (temp_path / "photo3.JPG").touch()
            (temp_path / "photo4.JPEG").touch()
            (temp_path / "not_photo.txt").touch()

            photos = find_jpeg_photos(temp_path)
            assert len(photos) == 4

            # Check that all found files are JPEG
            for photo in photos:
                assert photo.suffix.lower() in [".jpg", ".jpeg"]

    def test_find_jpeg_photos_in_subdirectories(self):
        """Test finding JPEG photos in subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create subdirectory structure
            subdir = temp_path / "subdir"
            subdir.mkdir()

            (temp_path / "photo1.jpg").touch()
            (subdir / "photo2.jpg").touch()

            photos = find_jpeg_photos(temp_path)
            assert len(photos) == 2

    def test_find_jpeg_photos_nonexistent_directory(self):
        """Test finding photos in a non-existent directory."""
        photos = find_jpeg_photos(Path("/nonexistent/directory"))
        assert photos == []


class TestGenerateXmpSidecar:
    """Tests for XMP sidecar file generation."""

    def test_generate_xmp_sidecar_basic(self):
        """Test basic XMP sidecar generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            photo_path = temp_path / "test_photo.jpg"
            photo_path.touch()

            analysis = PhotoAnalysis(
                photo_path=photo_path,
                rating=4,
                rating_reason="Good photo",
                keywords=["sunset", "beach", "ocean", "waves", "sky"],
                description="Beautiful sunset over the ocean with waves.",
            )

            xmp_path = generate_xmp_sidecar(analysis)

            # Check that XMP file was created
            assert xmp_path.exists()
            assert xmp_path.suffix == ".xmp"
            assert xmp_path.stem == "test_photo"

            # Parse and validate XMP content
            tree = ET.parse(xmp_path)
            root = tree.getroot()

            # Check namespaces
            assert "x:xmpmeta" in root.tag or "xmpmeta" in root.tag

            # Read content and check for expected values
            content = xmp_path.read_text()
            assert "xmp:Rating" in content
            assert "<xmp:Rating>4</xmp:Rating>" in content
            assert "dc:subject" in content
            assert "sunset" in content
            assert "beach" in content
            assert "dc:description" in content
            assert "Beautiful sunset over the ocean with waves." in content

    def test_generate_xmp_sidecar_custom_output_path(self):
        """Test XMP sidecar generation with custom output path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            photo_path = temp_path / "test_photo.jpg"
            photo_path.touch()

            custom_output = temp_path / "custom_name.xmp"

            analysis = PhotoAnalysis(
                photo_path=photo_path,
                rating=5,
                rating_reason="Excellent photo",
                keywords=["test", "photo", "sample", "demo", "example"],
                description="Test photo description.",
            )

            xmp_path = generate_xmp_sidecar(analysis, custom_output)

            # Check that custom path was used
            assert xmp_path == custom_output
            assert xmp_path.exists()

    def test_generate_xmp_sidecar_all_ratings(self):
        """Test XMP sidecar generation with different ratings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for rating in range(1, 6):
                photo_path = temp_path / f"photo_{rating}.jpg"
                photo_path.touch()

                analysis = PhotoAnalysis(
                    photo_path=photo_path,
                    rating=rating,
                    rating_reason="Test reason",
                    keywords=["test"] * 5,
                    description="Test description",
                )

                xmp_path = generate_xmp_sidecar(analysis)
                content = xmp_path.read_text()
                assert f"<xmp:Rating>{rating}</xmp:Rating>" in content


class TestPhotoAnalysis:
    """Tests for PhotoAnalysis dataclass."""

    def test_photo_analysis_creation(self):
        """Test creating a PhotoAnalysis object."""
        photo_path = Path("/test/photo.jpg")
        analysis = PhotoAnalysis(
            photo_path=photo_path,
            rating=3,
            rating_reason="Average photo",
            keywords=["test", "sample", "photo", "demo", "image"],
            description="A test photo",
        )

        assert analysis.photo_path == photo_path
        assert analysis.rating == 3
        assert len(analysis.keywords) == 5
        assert analysis.description == "A test photo"
        assert analysis.raw_response is None

    def test_photo_analysis_with_raw_response(self):
        """Test PhotoAnalysis with raw response."""
        analysis = PhotoAnalysis(
            photo_path=Path("/test/photo.jpg"),
            rating=4,
            rating_reason="Good photo",
            keywords=["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
            description="Description",
            raw_response="Raw VLM response",
        )

        assert analysis.raw_response == "Raw VLM response"

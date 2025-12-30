"""Tests for EXIF orientation handling in TPRS UI.

These tests validate that the EXIF orientation fix is correctly implemented
without requiring mlx_vlm dependencies.
"""

from pathlib import Path
import pytest


class TestExifOrientationHandling:
    """Tests for EXIF orientation handling in TPRS UI."""

    def test_tprs_ui_imports_imageops(self):
        """Test that tprs_ui.py imports ImageOps from PIL."""
        # Read the tprs_ui.py file and verify ImageOps is imported
        tprs_ui_path = Path(__file__).parent.parent / "src" / "tvas" / "tprs_ui.py"
        content = tprs_ui_path.read_text()
        
        # Check that ImageOps is imported from PIL
        assert "from PIL import Image, ImageDraw, ImageOps" in content or \
               "from PIL import ImageOps" in content, \
               "ImageOps should be imported from PIL in tprs_ui.py"
    
    def test_tprs_ui_uses_exif_transpose(self):
        """Test that tprs_ui.py uses ImageOps.exif_transpose when processing images."""
        # Read the tprs_ui.py file and verify exif_transpose is used
        tprs_ui_path = Path(__file__).parent.parent / "src" / "tvas" / "tprs_ui.py"
        content = tprs_ui_path.read_text()
        
        # Check that exif_transpose is called
        assert "ImageOps.exif_transpose" in content, \
               "ImageOps.exif_transpose should be called in tprs_ui.py"
        
        # Check that it's used in the show_details method context
        # Find the show_details method
        show_details_start = content.find("def show_details(")
        assert show_details_start != -1, "show_details method should exist"
        
        # Get the content after show_details
        remaining_content = content[show_details_start:]
        
        # Find the next method definition (or end of class)
        next_method = remaining_content.find("\n    def ", 10)
        if next_method == -1:
            show_details_content = remaining_content
        else:
            show_details_content = remaining_content[:next_method]
        
        # Verify exif_transpose is used in show_details
        assert "ImageOps.exif_transpose" in show_details_content, \
               "ImageOps.exif_transpose should be called in show_details method"
        
        # Verify it's called on the opened image
        assert "img = ImageOps.exif_transpose(img)" in show_details_content, \
               "Image should be processed with ImageOps.exif_transpose(img)"
    
    def test_exif_transpose_called_after_image_open(self):
        """Test that exif_transpose is called shortly after Image.open."""
        # Read the tprs_ui.py file
        tprs_ui_path = Path(__file__).parent.parent / "src" / "tvas" / "tprs_ui.py"
        content = tprs_ui_path.read_text()
        
        # Find the show_details method
        show_details_start = content.find("def show_details(")
        assert show_details_start != -1
        
        remaining_content = content[show_details_start:]
        next_method = remaining_content.find("\n    def ", 10)
        if next_method == -1:
            show_details_content = remaining_content
        else:
            show_details_content = remaining_content[:next_method]
        
        # Check that both Image.open and exif_transpose are present
        assert "Image.open" in show_details_content
        assert "ImageOps.exif_transpose" in show_details_content
        
        # Find the lines
        lines = show_details_content.split('\n')
        image_open_line = -1
        exif_transpose_line = -1
        
        for i, line in enumerate(lines):
            if 'Image.open' in line:
                image_open_line = i
            if 'ImageOps.exif_transpose' in line:
                exif_transpose_line = i
        
        assert image_open_line != -1, "Image.open should be found"
        assert exif_transpose_line != -1, "ImageOps.exif_transpose should be found"
        
        # exif_transpose should be called after Image.open but within a few lines
        assert exif_transpose_line > image_open_line, \
            "exif_transpose should be called after Image.open"
        assert exif_transpose_line - image_open_line <= 5, \
            "exif_transpose should be called shortly after Image.open (within 5 lines)"

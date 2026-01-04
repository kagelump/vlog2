import sys
from unittest.mock import patch
import argparse
import pytest
from pathlib import Path

# Since main() parses sys.argv, we need to mock it

def test_tvas_default_lmstudio_non_darwin():
    from tvas.main import main
    with patch("sys.platform", "linux"), \
         patch("sys.argv", ["tvas"]), \
         patch("tvas.main.TVASApp") as MockApp, \
         patch("tvas.main.find_camera_volumes", return_value=[]):
        try:
            main()
        except SystemExit:
            pass
        
        # Check if api_base was set to LM Studio default
        args, kwargs = MockApp.call_args
        assert kwargs["api_base"] == "http://localhost:1234/v1"

def test_tvas_no_default_lmstudio_darwin():
    from tvas.main import main
    with patch("sys.platform", "darwin"), \
         patch("sys.argv", ["tvas"]), \
         patch("tvas.main.TVASApp") as MockApp, \
         patch("tvas.main.find_camera_volumes", return_value=[]):
        try:
            main()
        except SystemExit:
            pass
        
        # Check if api_base was None (default)
        args, kwargs = MockApp.call_args
        assert kwargs["api_base"] is None

def test_tprs_default_lmstudio_non_darwin():
    from tprs.cli import main
    with patch("sys.platform", "linux"), \
         patch("sys.argv", ["tprs", "--headless", "/tmp"]), \
         patch("tprs.cli.process_photos_batch", return_value=[]) as MockBatch, \
         patch("tprs.cli.find_jpeg_photos", return_value=[Path("/tmp/test.jpg")]) as MockFind:
        try:
            main()
        except SystemExit:
            pass
        
        # Check if process_photos_batch was called with LM Studio model and api_base
        # In TPRS, if lmstudio is True, it sets model to "qwen/qwen3-vl"
        args, kwargs = MockBatch.call_args
        assert kwargs["api_base"] == "http://localhost:1234/v1"
        assert args[1] == "qwen/qwen3-vl"

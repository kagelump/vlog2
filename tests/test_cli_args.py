import argparse
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from tprs import cli as tprs_cli

def test_openrouter_headless_args():
    """Test --openrouter in headless mode."""
    with patch('sys.argv', ['tprs', '.', '--headless', '--openrouter']), \
         patch('tprs.cli.check_lmstudio_running', return_value=False), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.read_text', return_value="sk-test-key"), \
         patch('tprs.cli.find_jpeg_photos', return_value=['photo.jpg']), \
         patch('tprs.cli.process_photos_batch') as mock_process:
        
        # Mock Path.home() to avoid reading real file
        with patch('pathlib.Path.home', return_value=Path('/tmp')):
             # We need to mock the specific file check
             with patch('pathlib.Path.exists', side_effect=lambda: True): # For directory and key file
                 with patch('pathlib.Path.read_text', return_value="sk-test-key"):
                     tprs_cli.main()
        
        mock_process.assert_called_once()
        call_args = mock_process.call_args
        assert call_args[1]['api_base'] == "https://openrouter.ai/api/v1"
        assert call_args[1]['api_key'] == "sk-test-key"


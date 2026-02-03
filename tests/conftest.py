"""
pytest configuration file to ensure src directory is in sys.path
"""
import sys
from pathlib import Path

# Add the src directory to sys.path so that modules can be imported
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

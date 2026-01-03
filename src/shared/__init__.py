"""Shared utilities for TVAS and TPRS.

Common code used by both Travel Vlog Automation System and Travel Photo Rating System.
"""

import logging
import os
from pathlib import Path

__version__ = "0.1.0"

# Default model for mlx-vlm
DEFAULT_VLM_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-8bit"

_PROMPT_OVERRIDES = {}

logger = logging.getLogger(__name__)

def set_prompt_override(filename: str, content: str):
    """Set an override for a prompt file."""
    _PROMPT_OVERRIDES[filename] = content

def get_prompt_override(filename: str) -> str | None:
    """Get the current override for a prompt file, if any."""
    return _PROMPT_OVERRIDES.get(filename)

def load_prompt(filename: str) -> str:
    """Load a prompt from the prompts directory or overrides."""
    if filename in _PROMPT_OVERRIDES:
        return _PROMPT_OVERRIDES[filename]

    prompts_dir = Path(__file__).parent / "prompts"
    prompt_path = prompts_dir / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text().strip()


def get_openrouter_api_key(api_key: str | None = None) -> str | None:
    """Get OpenRouter API key from argument, environment, or file."""
    if api_key:
        return api_key
    
    if "OPENROUTER_API_KEY" in os.environ:
        return os.environ["OPENROUTER_API_KEY"]
    
    key_path = Path.home() / ".openrouterkey"
    if key_path.exists():
        try:
            key = key_path.read_text().strip()
            logger.info(f"Loaded OpenRouter API key from {key_path}")
            return key
        except Exception as e:
            logger.warning(f"Failed to read {key_path}: {e}")
    
    logger.warning("OpenRouter mode enabled but no API key found in env or ~/.openrouterkey")
    return None

"""Travel Vlog Automation System (TVAS)

Automate vlog ingestion, junk detection, and DaVinci Resolve import.
"""

from pathlib import Path

__version__ = "0.1.0"

# Default model for mlx-vlm
DEFAULT_VLM_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-8bit"

_PROMPT_OVERRIDES = {}

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

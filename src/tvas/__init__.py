"""Travel Vlog Automation System (TVAS)

Automate vlog ingestion, junk detection, and DaVinci Resolve import.
"""

from pathlib import Path

__version__ = "0.1.0"

# Default model for mlx-vlm
DEFAULT_VLM_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-8bit"

def load_prompt(filename: str) -> str:
    """Load a prompt from the prompts directory."""
    prompts_dir = Path(__file__).parent / "prompts"
    prompt_path = prompts_dir / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text().strip()

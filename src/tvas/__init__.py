"""Travel Vlog Automation System (TVAS)

Automate vlog ingestion, junk detection, and DaVinci Resolve import.
"""

from shared import DEFAULT_VLM_MODEL, load_prompt, set_prompt_override, get_prompt_override

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_VLM_MODEL",
    "load_prompt",
    "set_prompt_override",
    "get_prompt_override",
]

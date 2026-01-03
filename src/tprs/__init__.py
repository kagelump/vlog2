"""Travel Photo Rating System (TPRS)

Analyze photos using Qwen3 VL and generate XMP sidecar files with ratings,
keywords, and descriptions.
"""

from tprs.tprs import (
    PhotoAnalysis,
    find_jpeg_photos,
    process_photos_batch,
)

__all__ = [
    "PhotoAnalysis",
    "find_jpeg_photos",
    "process_photos_batch",
]

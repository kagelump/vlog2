from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def detect_archival_root(explicit_path: Path | None) -> Path | None:
    """Detect the archival storage root path."""
    if explicit_path:
        return explicit_path
    
    # Check default locations
    defaults = [Path("/Volumes/Acasis")]
    for path in defaults:
        if path.exists():
            return path
    return None


def find_latest_project(archival_root: Path) -> Path | None:
    """Find the most recently modified project directory in archival root."""
    if not archival_root or not archival_root.exists():
        return None
        
    try:
        project_dirs = sorted(
            [d for d in archival_root.iterdir() if d.is_dir() and not d.name.startswith('.')],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        return project_dirs[0] if project_dirs else None
    except Exception as e:
        logger.warning(f"Failed to list projects in {archival_root}: {e}")
        return None


def resolve_project_path(
    archival_root: Path | None, 
    project_name_arg: str | None, 
    analysis_path_arg: Path | None = None
) -> Path | None:
    """Resolve the project directory path."""
    # 1. Use analysis path if it exists (treated as project path in CLI if compatible)
    if analysis_path_arg and analysis_path_arg.exists():
        return analysis_path_arg
        
    # 2. Use specific project name in archival root
    if project_name_arg and archival_root and archival_root.exists():
        return archival_root / project_name_arg
        
    # 3. Fallback to latest project
    if archival_root and archival_root.exists():
        return find_latest_project(archival_root)
        
    return None

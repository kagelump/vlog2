"""Volume Watcher

This module monitors for SD card/volume insertions using the watchdog library.
"""

import logging
import platform
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Watchdog is optional for volume monitoring
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object
    Observer = None


def check_watchdog_available() -> bool:
    """Check if watchdog is available.

    Returns:
        True if watchdog is available.
    """
    return WATCHDOG_AVAILABLE


def get_volumes_path() -> Path:
    """Get the path where volumes are mounted based on OS.

    Returns:
        Path to volumes directory.
    """
    system = platform.system()
    if system == "Darwin":  # macOS
        return Path("/Volumes")
    elif system == "Linux":
        return Path("/media") / (Path.home().name)
    else:
        # Windows or unknown - default to current user's path
        return Path.home()


class VolumeEventHandler(FileSystemEventHandler):
    """Handler for volume mount/unmount events."""

    def __init__(
        self,
        on_volume_added: Callable[[Path], None] | None = None,
        on_volume_removed: Callable[[Path], None] | None = None,
    ):
        """Initialize the handler.

        Args:
            on_volume_added: Callback when a volume is added.
            on_volume_removed: Callback when a volume is removed.
        """
        super().__init__()
        self.on_volume_added = on_volume_added
        self.on_volume_removed = on_volume_removed
        self._known_volumes: set[str] = set()

        # Initialize with existing volumes
        self._scan_existing_volumes()

    def _scan_existing_volumes(self):
        """Scan for existing volumes."""
        volumes_path = get_volumes_path()
        if volumes_path.exists():
            for item in volumes_path.iterdir():
                if item.is_dir():
                    self._known_volumes.add(str(item))

    def on_created(self, event):
        """Handle directory creation (volume mount)."""
        if event.is_directory:
            path = Path(event.src_path)
            if str(path) not in self._known_volumes:
                self._known_volumes.add(str(path))
                logger.info(f"Volume mounted: {path}")
                if self.on_volume_added:
                    self.on_volume_added(path)

    def on_deleted(self, event):
        """Handle directory deletion (volume unmount)."""
        if event.is_directory:
            path = Path(event.src_path)
            if str(path) in self._known_volumes:
                self._known_volumes.discard(str(path))
                logger.info(f"Volume unmounted: {path}")
                if self.on_volume_removed:
                    self.on_volume_removed(path)


class VolumeWatcher:
    """Watches for volume mounts and unmounts."""

    def __init__(
        self,
        on_volume_added: Callable[[Path], None] | None = None,
        on_volume_removed: Callable[[Path], None] | None = None,
    ):
        """Initialize the watcher.

        Args:
            on_volume_added: Callback when a volume is added.
            on_volume_removed: Callback when a volume is removed.
        """
        self.on_volume_added = on_volume_added
        self.on_volume_removed = on_volume_removed
        self._observer: Observer | None = None
        self._handler: VolumeEventHandler | None = None

    def start(self) -> bool:
        """Start watching for volume changes.

        Returns:
            True if started successfully.
        """
        if not WATCHDOG_AVAILABLE:
            logger.error("Watchdog not available - cannot watch for volumes")
            return False

        volumes_path = get_volumes_path()
        if not volumes_path.exists():
            logger.error(f"Volumes path does not exist: {volumes_path}")
            return False

        self._handler = VolumeEventHandler(
            on_volume_added=self.on_volume_added,
            on_volume_removed=self.on_volume_removed,
        )

        self._observer = Observer()
        self._observer.schedule(self._handler, str(volumes_path), recursive=False)
        self._observer.start()

        logger.info(f"Started watching {volumes_path} for volume changes")
        return True

    def stop(self):
        """Stop watching for volume changes."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Stopped volume watcher")

    def is_running(self) -> bool:
        """Check if the watcher is running.

        Returns:
            True if watching.
        """
        return self._observer is not None and self._observer.is_alive()


def list_current_volumes() -> list[Path]:
    """List all currently mounted volumes.

    Returns:
        List of volume paths.
    """
    volumes_path = get_volumes_path()
    volumes = []

    if volumes_path.exists():
        for item in volumes_path.iterdir():
            if item.is_dir():
                volumes.append(item)

    return volumes


def is_camera_volume(volume_path: Path) -> bool:
    """Check if a volume appears to be a camera storage.

    Args:
        volume_path: Path to the volume.

    Returns:
        True if the volume appears to be camera storage.
    """
    from tvas.ingestion import CameraType, detect_camera_type

    camera_type = detect_camera_type(volume_path)
    return camera_type != CameraType.UNKNOWN


def find_camera_volumes() -> list[Path]:
    """Find all currently mounted camera volumes.

    Returns:
        List of camera volume paths.
    """
    volumes = list_current_volumes()
    camera_volumes = []

    for volume in volumes:
        if is_camera_volume(volume):
            camera_volumes.append(volume)
            logger.info(f"Found camera volume: {volume}")

    return camera_volumes

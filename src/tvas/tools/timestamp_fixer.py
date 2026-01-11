"""Timestamp Fixer Tool for TVAS

This module provides tools for detecting and correcting timestamp inconsistencies
across video files from multiple cameras. Supports detecting anomalies within
camera filesets and applying time shifts to selected subsets of files.
"""

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from shared.proxy import get_video_duration

logger = logging.getLogger(__name__)

# Video file extensions to process
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mts", ".m4v", ".avi", ".mkv"}

# Threshold for detecting timestamp anomalies (jumps > this are flagged)
ANOMALY_THRESHOLD_HOURS = 6


@dataclass
class ClipInfo:
    """Information about a single video clip."""
    
    path: Path
    camera: str
    created_at: datetime
    duration_seconds: float
    file_size: int
    is_anomaly: bool = False
    anomaly_gap_hours: float = 0.0
    
    @property
    def end_time(self) -> datetime:
        """Calculate end time based on start + duration."""
        return self.created_at + timedelta(seconds=self.duration_seconds)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "path": str(self.path),
            "camera": self.camera,
            "created_at": self.created_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "file_size": self.file_size,
            "is_anomaly": self.is_anomaly,
            "anomaly_gap_hours": self.anomaly_gap_hours,
        }


@dataclass
class ShiftOperation:
    """Record of a time shift operation for undo/redo."""
    
    file_paths: list[Path]
    delta: timedelta
    original_times: dict[Path, datetime]  # Maps path -> original created_at
    applied_at: datetime = field(default_factory=datetime.now)
    
    @property
    def description(self) -> str:
        """Human-readable description of the operation."""
        sign = "+" if self.delta.total_seconds() >= 0 else ""
        hours, remainder = divmod(abs(self.delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{sign}{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} to {len(self.file_paths)} files"


class TimeShiftEngine:
    """Engine for managing timestamp corrections across video files.
    
    Handles loading project files, detecting anomalies, and applying
    time shifts with undo/redo support.
    """
    
    def __init__(self):
        self.clips: list[ClipInfo] = []
        self.cameras: dict[str, list[ClipInfo]] = {}  # camera_name -> clips
        self.undo_stack: list[ShiftOperation] = []
        self.redo_stack: list[ShiftOperation] = []
        self.project_path: Optional[Path] = None
        
    def load_project(
        self, 
        path: Path, 
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> None:
        """Scan a project directory for video files.
        
        Args:
            path: Path to project root (should contain ingested/ or archival/).
            progress_callback: Optional callback(current, total, message).
        """
        self.project_path = path
        self.clips = []
        self.cameras = {}
        self.undo_stack = []
        self.redo_stack = []
        
        # Find video directories
        search_dirs = []
        for subdir in ["ingested", "archival", "proxy"]:
            subpath = path / subdir
            if subpath.exists():
                search_dirs.append(subpath)
        
        if not search_dirs:
            # Assume path itself contains videos
            search_dirs = [path]
        
        # Collect all video files
        video_files: list[tuple[Path, str]] = []  # (path, camera_name)
        
        for search_dir in search_dirs:
            for video_path in search_dir.rglob("*"):
                if video_path.name.startswith("."):
                    continue
                
                if video_path.suffix.lower() in VIDEO_EXTENSIONS and video_path.is_file():
                    # Determine camera name from parent folder structure
                    # Expected: ingested/YYYY-MM-DD/CameraName/file.mp4
                    # or: archival/CameraName/YYYY-MM-DD/file.mp4
                    camera_name = self._detect_camera_from_path(video_path, search_dir)
                    video_files.append((video_path, camera_name))
        
        total = len(video_files)
        logger.info(f"Found {total} video files to process")
        
        for idx, (video_path, camera_name) in enumerate(video_files):
            if progress_callback:
                progress_callback(idx + 1, total, f"Loading {video_path.name}")
            
            try:
                clip = self._load_clip_info(video_path, camera_name)
                self.clips.append(clip)
                
                if camera_name not in self.cameras:
                    self.cameras[camera_name] = []
                self.cameras[camera_name].append(clip)
            except Exception as e:
                logger.warning(f"Failed to load {video_path}: {e}")
        
        # Sort clips within each camera by timestamp
        for camera_clips in self.cameras.values():
            camera_clips.sort(key=lambda c: c.created_at)
        
        # Sort all clips globally
        self.clips.sort(key=lambda c: c.created_at)
        
        # Detect anomalies
        self.detect_anomalies()
        
        logger.info(f"Loaded {len(self.clips)} clips from {len(self.cameras)} cameras")
    
    def _detect_camera_from_path(self, video_path: Path, search_dir: Path) -> str:
        """Detect camera name from the file path structure."""
        try:
            relative = video_path.relative_to(search_dir)
            parts = relative.parts
            
            # Look for camera-like folder names
            # Skip date-like folders (YYYY-MM-DD or YYYYMMDD)
            for part in parts[:-1]:  # Exclude filename
                if not self._is_date_folder(part):
                    return part
            
            # Fallback: use parent folder name
            return video_path.parent.name
        except ValueError:
            return video_path.parent.name
    
    def _is_date_folder(self, name: str) -> bool:
        """Check if folder name looks like a date."""
        # Common date patterns
        import re
        patterns = [
            r"^\d{4}-\d{2}-\d{2}$",  # YYYY-MM-DD
            r"^\d{8}$",              # YYYYMMDD
            r"^\d{4}_\d{2}_\d{2}$",  # YYYY_MM_DD
        ]
        return any(re.match(p, name) for p in patterns)
    
    def _load_clip_info(self, path: Path, camera: str) -> ClipInfo:
        """Load metadata for a single video clip."""
        stat = path.stat()
        
        # Get creation time (prefer birthtime on macOS)
        if hasattr(stat, "st_birthtime"):
            created_ts = stat.st_birthtime
        else:
            created_ts = stat.st_ctime
        
        created_at = datetime.fromtimestamp(created_ts)
        
        # Get duration using ffprobe
        try:
            duration = get_video_duration(path)
            if duration is None:
                duration = 0.0
        except Exception:
            duration = 0.0
        
        return ClipInfo(
            path=path,
            camera=camera,
            created_at=created_at,
            duration_seconds=duration,
            file_size=stat.st_size,
        )
    
    def detect_anomalies(self, threshold_hours: float = ANOMALY_THRESHOLD_HOURS) -> list[ClipInfo]:
        """Detect timestamp anomalies within each camera's fileset.
        
        An anomaly is a gap > threshold_hours between consecutive clips
        from the same camera.
        
        Args:
            threshold_hours: Minimum gap to flag as anomaly.
            
        Returns:
            List of clips that are flagged as anomalies.
        """
        anomalies = []
        
        for camera_name, camera_clips in self.cameras.items():
            if len(camera_clips) < 2:
                continue
            
            # Sort by timestamp
            sorted_clips = sorted(camera_clips, key=lambda c: c.created_at)
            
            for i in range(1, len(sorted_clips)):
                prev_clip = sorted_clips[i - 1]
                curr_clip = sorted_clips[i]
                
                # Calculate gap (accounting for duration of previous clip)
                expected_end = prev_clip.created_at + timedelta(seconds=prev_clip.duration_seconds)
                gap = (curr_clip.created_at - expected_end).total_seconds() / 3600
                
                if abs(gap) > threshold_hours:
                    curr_clip.is_anomaly = True
                    curr_clip.anomaly_gap_hours = gap
                    anomalies.append(curr_clip)
                    logger.info(
                        f"Anomaly detected: {curr_clip.path.name} has {gap:.1f}h gap "
                        f"after {prev_clip.path.name} in {camera_name}"
                    )
        
        return anomalies
    
    def get_anomalies(self) -> list[ClipInfo]:
        """Get all clips flagged as anomalies."""
        return [c for c in self.clips if c.is_anomaly]
    
    def get_time_range(self) -> tuple[datetime, datetime]:
        """Get the overall time range of all clips."""
        if not self.clips:
            now = datetime.now()
            return now, now
        
        min_time = min(c.created_at for c in self.clips)
        max_time = max(c.end_time for c in self.clips)
        return min_time, max_time
    
    def calculate_sync_offset(self, clip_a: ClipInfo, clip_b: ClipInfo) -> timedelta:
        """Calculate the time offset needed to align clip_b to clip_a.
        
        Returns:
            Timedelta to add to clip_b's timestamp to align it with clip_a.
        """
        return clip_a.created_at - clip_b.created_at
    
    def apply_shift(
        self, 
        clips: list[ClipInfo], 
        delta: timedelta,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> bool:
        """Apply a time shift to the selected clips.
        
        Args:
            clips: List of clips to shift.
            delta: Time offset to apply (positive = later, negative = earlier).
            progress_callback: Optional callback(current, total, message).
            
        Returns:
            True if successful, False otherwise.
        """
        if not clips:
            return False
        
        # Record original times for undo
        original_times = {c.path: c.created_at for c in clips}
        
        total = len(clips)
        failed = []
        
        for idx, clip in enumerate(clips):
            if progress_callback:
                progress_callback(idx + 1, total, f"Updating {clip.path.name}")
            
            new_time = clip.created_at + delta
            
            try:
                self._update_file_timestamp(clip.path, new_time)
                clip.created_at = new_time
            except Exception as e:
                logger.error(f"Failed to update {clip.path}: {e}")
                failed.append(clip.path)
        
        if failed:
            logger.warning(f"Failed to update {len(failed)} files")
        
        # Record operation for undo
        operation = ShiftOperation(
            file_paths=[c.path for c in clips],
            delta=delta,
            original_times=original_times,
        )
        self.undo_stack.append(operation)
        self.redo_stack.clear()  # Clear redo stack on new operation
        
        # Re-sort clips
        self.clips.sort(key=lambda c: c.created_at)
        for camera_clips in self.cameras.values():
            camera_clips.sort(key=lambda c: c.created_at)
        
        # Re-detect anomalies
        for clip in self.clips:
            clip.is_anomaly = False
            clip.anomaly_gap_hours = 0.0
        self.detect_anomalies()
        
        return len(failed) == 0
    
    def _update_file_timestamp(self, path: Path, new_time: datetime) -> None:
        """Update a file's creation and modification timestamps.
        
        On macOS, uses SetFile to update creation time (st_birthtime).
        Falls back to os.utime for modification time.
        """
        timestamp = new_time.timestamp()
        
        if platform.system() == "Darwin":
            # macOS: Use SetFile to update creation date
            # Format: "mm/dd/yyyy HH:MM:SS"
            date_str = new_time.strftime("%m/%d/%Y %H:%M:%S")
            
            try:
                result = subprocess.run(
                    ["SetFile", "-d", date_str, str(path)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.warning(f"SetFile failed: {result.stderr}")
            except FileNotFoundError:
                logger.warning(
                    "SetFile not found. Install Xcode Command Line Tools: "
                    "xcode-select --install"
                )
        
        # Update modification time (works on all platforms)
        os.utime(path, (timestamp, timestamp))
    
    def undo(self) -> Optional[ShiftOperation]:
        """Undo the last shift operation.
        
        Returns:
            The undone operation, or None if nothing to undo.
        """
        if not self.undo_stack:
            return None
        
        operation = self.undo_stack.pop()
        
        # Restore original times
        for clip in self.clips:
            if clip.path in operation.original_times:
                original_time = operation.original_times[clip.path]
                try:
                    self._update_file_timestamp(clip.path, original_time)
                    clip.created_at = original_time
                except Exception as e:
                    logger.error(f"Failed to undo {clip.path}: {e}")
        
        self.redo_stack.append(operation)
        
        # Re-sort and re-detect
        self.clips.sort(key=lambda c: c.created_at)
        for camera_clips in self.cameras.values():
            camera_clips.sort(key=lambda c: c.created_at)
        
        for clip in self.clips:
            clip.is_anomaly = False
            clip.anomaly_gap_hours = 0.0
        self.detect_anomalies()
        
        return operation
    
    def redo(self) -> Optional[ShiftOperation]:
        """Redo a previously undone operation.
        
        Returns:
            The redone operation, or None if nothing to redo.
        """
        if not self.redo_stack:
            return None
        
        operation = self.redo_stack.pop()
        
        # Re-apply the shift
        for clip in self.clips:
            if clip.path in operation.original_times:
                new_time = operation.original_times[clip.path] + operation.delta
                try:
                    self._update_file_timestamp(clip.path, new_time)
                    clip.created_at = new_time
                except Exception as e:
                    logger.error(f"Failed to redo {clip.path}: {e}")
        
        self.undo_stack.append(operation)
        
        # Re-sort and re-detect
        self.clips.sort(key=lambda c: c.created_at)
        for camera_clips in self.cameras.values():
            camera_clips.sort(key=lambda c: c.created_at)
        
        for clip in self.clips:
            clip.is_anomaly = False
            clip.anomaly_gap_hours = 0.0
        self.detect_anomalies()
        
        return operation
    
    def get_clips_by_camera(self, camera: str) -> list[ClipInfo]:
        """Get all clips from a specific camera."""
        return self.cameras.get(camera, [])
    
    def get_clips_in_range(
        self, 
        start: datetime, 
        end: datetime, 
        cameras: Optional[list[str]] = None
    ) -> list[ClipInfo]:
        """Get clips within a time range, optionally filtered by cameras."""
        result = []
        for clip in self.clips:
            if cameras and clip.camera not in cameras:
                continue
            if start <= clip.created_at <= end:
                result.append(clip)
        return result
    
    def parse_time_delta(self, time_str: str) -> Optional[timedelta]:
        """Parse a time string like '+01:30:00' or '-00:05:00' into a timedelta.
        
        Args:
            time_str: Time string in format [+/-]HH:MM:SS or [+/-]HH:MM
            
        Returns:
            Parsed timedelta, or None if invalid.
        """
        time_str = time_str.strip()
        if not time_str:
            return None
        
        # Handle sign
        negative = time_str.startswith("-")
        if time_str.startswith(("+", "-")):
            time_str = time_str[1:]
        
        parts = time_str.split(":")
        try:
            if len(parts) == 2:
                hours, minutes = int(parts[0]), int(parts[1])
                seconds = 0
            elif len(parts) == 3:
                hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                return None
            
            total_seconds = hours * 3600 + minutes * 60 + seconds
            if negative:
                total_seconds = -total_seconds
            
            return timedelta(seconds=total_seconds)
        except ValueError:
            return None
    
    def format_time_delta(self, delta: timedelta) -> str:
        """Format a timedelta as a string like '+01:30:00'."""
        total_seconds = int(delta.total_seconds())
        sign = "+" if total_seconds >= 0 else "-"
        total_seconds = abs(total_seconds)
        
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"

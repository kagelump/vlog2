"""Main entry point for TVAS (Travel Vlog Automation System).

This module orchestrates the complete workflow from SD card detection
to timeline generation.
"""

import argparse
import logging
import signal
import sys
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from shared import DEFAULT_VLM_MODEL, get_openrouter_api_key
from tvas.analysis import analyze_clips_batch
from tvas.ingestion import CameraType, detect_camera_type, ingest_volume
from shared.proxy import generate_proxies_batch
from tvas.timeline import TimelineConfig, create_timeline_from_analysis, export_analysis_json
from tvas.watcher import VolumeWatcher, check_watchdog_available, find_camera_volumes, is_camera_volume

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TVASApp:
    """Main TVAS application."""

    def __init__(
        self,
        archival_path: Path | None = None,
        proxy_path: Path | None = None,
        use_vlm: bool = True,
        vlm_model: str = DEFAULT_VLM_MODEL,
        auto_approve: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        provider_preferences: str | None = None,
    ):
        """Initialize the TVAS application.

        Args:
            archival_path: Path for archival storage (SSD). Auto-detects /Volumes/ACASIS if None.
            proxy_path: Path for local proxy storage (default: ~/Movies/Vlog).
            use_vlm: Whether to use VLM for analysis.
            vlm_model: mlx-vlm model name for VLM.
            auto_approve: Auto-approve all AI decisions without UI.
            api_base: Base URL for VLM API (e.g. OpenRouter or LM Studio).
            api_key: API key for VLM API.
            provider_preferences: Comma-separated list of preferred providers (for OpenRouter).
        """
        # Auto-detect ACASIS volume for archival
        if archival_path is None:
            acasis_path = Path("/Volumes/Acasis")
            if acasis_path.exists():
                self.archival_path = acasis_path
                logger.info(f"Auto-detected archival storage: {acasis_path}")
            else:
                self.archival_path = None
                logger.info("No archival storage found - will skip file copying")
        else:
            self.archival_path = archival_path
        
        self.proxy_path = proxy_path or Path.home() / "Movies" / "Vlog"
        self.use_vlm = use_vlm
        self.vlm_model = vlm_model
        self.auto_approve = auto_approve
        self.api_base = api_base
        self.api_key = api_key
        self.provider_preferences = provider_preferences
        self.watcher: VolumeWatcher | None = None
        self._running = False

    def _process_pipeline(
        self,
        source_files: list[Path],
        project_name: str,
    ) -> dict[str, Any]:
        """Run the processing pipeline on a list of source files.

        Args:
            source_files: List of video file paths to process.
            project_name: Project name for organization.

        Returns:
            Dictionary with processing results.
        """
        results: dict[str, Any] = {
            "success": False,
            "files_processed": len(source_files),
            "clips_analyzed": 0,
            "timeline_path": None,
            "errors": [],
        }

        if not source_files:
            results["errors"].append("No files to process")
            return results

        proxy_dir = self.proxy_path / project_name / "proxy"

        # Stage 2: Generate edit proxies
        logger.info("Stage 2: Generating edit proxies...")
        edit_proxy_results = []
        try:
            edit_proxy_results = generate_proxies_batch(
                source_files,
                proxy_dir,
            )
            successful_edit_proxies = [r for r in edit_proxy_results if r.success]
            logger.info(f"Generated {len(successful_edit_proxies)}/{len(edit_proxy_results)} edit proxies")
        except Exception as e:
            results["errors"].append(f"Edit proxy generation failed: {e}")
            logger.error(f"Edit proxy generation failed: {e}")

        # Stage 3: AI Analysis (uses edit proxies)
        logger.info("Stage 3: Analyzing clips...")
        clips_to_analyze: list[tuple[Path, Path | None]] = [
            (r.source_path, r.proxy_path) for r in edit_proxy_results if r.success and r.proxy_path
        ]

        # Add any existing proxies in the directory that weren't just generated/processed
        # This allows for cumulative processing/export
        if proxy_dir.exists():
            current_proxies = {p for _, p in clips_to_analyze if p}
            for proxy_file in proxy_dir.glob("*.mp4"):
                if proxy_file not in current_proxies:
                    logger.info(f"Found existing proxy: {proxy_file.name}")
                    # For existing proxies, we use the proxy path as source if we can't determine the original
                    clips_to_analyze.append((proxy_file, proxy_file))
        try:
            analyses = analyze_clips_batch(
                clips_to_analyze,
                use_vlm=self.use_vlm,
                model_name=self.vlm_model,
                api_base=self.api_base,
                api_key=self.api_key,
                provider_preferences=self.provider_preferences,
            )
            results["clips_analyzed"] = len(analyses)

            # Export analysis for debugging
            analysis_json = self.proxy_path / project_name / "analysis.json"
            export_analysis_json(analyses, analysis_json)
            logger.info(f"Analysis exported to {analysis_json}")
        except Exception as e:
            results["errors"].append(f"Analysis failed: {e}")
            logger.error(f"Analysis failed: {e}")
            raise e

        # Stage 4: Generate Timeline (user review happens in DaVinci Resolve)
        logger.info("Stage 4: Generating timeline...")
        timeline_path = (
            self.proxy_path
            / project_name
            / f"{project_name}_timeline.otio"
        )

        try:
            result_path = create_timeline_from_analysis(
                analyses,
                timeline_path,
                TimelineConfig(name=project_name),
            )
            if result_path:
                results["timeline_path"] = str(result_path)
                results["success"] = True
                logger.info(f"Timeline created: {result_path}")
            else:
                results["errors"].append("Timeline generation returned None")
        except Exception as e:
            results["errors"].append(f"Timeline generation failed: {e}")
            logger.error(f"Timeline generation failed: {e}")

        return results

    def process_volume(
        self,
        volume_path: Path,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        """Process a single volume through the complete pipeline.

        Args:
            volume_path: Path to the mounted volume.
            project_name: Optional project name (default: date-based).

        Returns:
            Dictionary with processing results.
        """
        results: dict[str, Any] = {
            "volume": str(volume_path),
            "success": False,
            "files_processed": 0,
            "clips_analyzed": 0,
            "timeline_path": None,
            "errors": [],
        }

        # Detect camera type
        camera_type = detect_camera_type(volume_path)
        if camera_type == CameraType.UNKNOWN:
            results["errors"].append("Unknown camera type - skipping")
            logger.warning(f"Unknown camera type for {volume_path}")
            return results

        logger.info(f"Detected camera: {camera_type.value}")

        # Generate project name if not provided
        if project_name is None:
            project_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Stage 1: Ingest files (optional - skip if no archival path)
        source_files = []
        if self.archival_path:
            logger.info("Stage 1: Ingesting files to archival storage...")
            try:
                session = ingest_volume(
                    volume_path,
                    self.archival_path,
                    project_name,
                    progress_callback=lambda name, i, total: logger.info(
                        f"Copying {i}/{total}: {name}"
                    ),
                )
                logger.info(f"Ingested {len(session.files)} files")
                source_files = [f.destination_path for f in session.files if f.destination_path]
            except Exception as e:
                results["errors"].append(f"Ingestion failed: {e}")
                logger.error(f"Ingestion failed: {e}")
                return results

            if not session.files:
                results["errors"].append("No files found to process")
                return results
        else:
            # No archival - work directly from SD card
            logger.info("Stage 1: Skipping archival copy (no archival path)...")
            from tvas.ingestion import get_video_files
            video_files = get_video_files(volume_path, camera_type)
            source_files = [f.source_path for f in video_files]
            logger.info(f"Found {len(source_files)} files on SD card")
            
            if not source_files:
                results["errors"].append("No files found to process")
                return results

        # Run the rest of the pipeline
        pipeline_results = self._process_pipeline(source_files, project_name)
        results.update(pipeline_results)
        return results

    def process_from_archival(
        self,
        archival_dir: Path,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        """Process files from archival storage (skip ingestion stage).

        Args:
            archival_dir: Path to the archival directory containing camera folders.
            project_name: Optional project name (default: use directory name).

        Returns:
            Dictionary with processing results.
        """
        # Use directory name as project name if not provided
        if project_name is None:
            project_name = archival_dir.name

        logger.info(f"Processing from archival directory: {archival_dir}")

        # Gather all video files from camera subdirectories
        source_files = []
        camera_dirs = [d for d in archival_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if not camera_dirs:
            return {
                "archival_dir": str(archival_dir),
                "success": False,
                "errors": [f"No camera directories found in {archival_dir}"],
            }

        # Collect video files from all camera folders
        video_extensions = {'.mp4', '.MP4', '.mov', '.MOV', '.mts', '.MTS', '.insv', '.INSV', '.insp', '.INSP'}
        for camera_dir in camera_dirs:
            logger.info(f"Scanning {camera_dir.name}...")
            for video_file in camera_dir.iterdir():
                if video_file.is_file() and video_file.suffix in video_extensions:
                    source_files.append(video_file)
        
        if not source_files:
            return {
                "archival_dir": str(archival_dir),
                "success": False,
                "errors": ["No video files found in archival directory"],
            }

        logger.info(f"Found {len(source_files)} video files in archival storage")

        # Run the pipeline
        return self._process_pipeline(source_files, project_name)

    def process_directory(self, directory: Path) -> dict[str, Any]:
        """Process a directory of video files (analysis only).

        Args:
            directory: Path to the directory containing video files.

        Returns:
            Dictionary with processing results.
        """
        results: dict[str, Any] = {
            "directory": str(directory),
            "success": False,
            "files_processed": 0,
            "clips_analyzed": 0,
            "errors": [],
        }

        logger.info(f"Scanning directory: {directory}")

        # Gather all video files
        source_files = []
        video_extensions = {'.mp4', '.MP4', '.mov', '.MOV', '.mxf', '.MXF', '.mts', '.MTS', '.insv', '.INSV', '.insp', '.INSP'}
        
        for video_file in directory.iterdir():
            if video_file.is_file() and video_file.suffix in video_extensions:
                source_files.append(video_file)
        
        if not source_files:
            results["errors"].append(f"No video files found in {directory}")
            return results

        logger.info(f"Found {len(source_files)} video files")
        results["files_processed"] = len(source_files)

        # Analyze clips (pass None for proxy_path to analyze source directly)
        clips_to_analyze = [(f, None) for f in source_files]
        
        try:
            analyses = analyze_clips_batch(
                clips_to_analyze,
                use_vlm=self.use_vlm,
                model_name=self.vlm_model,
                api_base=self.api_base,
                api_key=self.api_key,
                provider_preferences=self.provider_preferences,
            )
            results["clips_analyzed"] = len(analyses)
            results["success"] = True
            
        except Exception as e:
            results["errors"].append(f"Analysis failed: {e}")
            logger.error(f"Analysis failed: {e}")

        return results

    def _on_volume_added(self, volume_path: Path):
        """Handle volume mount event."""
        logger.info(f"Volume added: {volume_path}")

        if is_camera_volume(volume_path):
            logger.info(f"Camera detected on {volume_path}")
            # In a real implementation, we'd show a notification and wait for user
            # For now, we'll just log it
            camera_type = detect_camera_type(volume_path)
            logger.info(f"Ready to process {camera_type.value} from {volume_path}")

    def _on_volume_removed(self, volume_path: Path):
        """Handle volume unmount event."""
        logger.info(f"Volume removed: {volume_path}")

    def start_watching(self) -> bool:
        """Start watching for volume changes.

        Returns:
            True if started successfully.
        """
        if not check_watchdog_available():
            logger.error("Watchdog not available - cannot watch for volumes")
            return False

        self.watcher = VolumeWatcher(
            on_volume_added=self._on_volume_added,
            on_volume_removed=self._on_volume_removed,
        )

        if self.watcher.start():
            self._running = True
            logger.info("TVAS is now watching for SD cards...")
            return True
        return False

    def stop_watching(self):
        """Stop watching for volume changes."""
        if self.watcher:
            self.watcher.stop()
            self._running = False
            logger.info("TVAS stopped watching")

    def run_daemon(self):
        """Run as a daemon, watching for volumes."""
        if not self.start_watching():
            return

        # Handle signals for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            self.stop_watching()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("TVAS daemon running. Press Ctrl+C to stop.")

        # Keep running
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_watching()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Travel Vlog Automation System (TVAS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tvas                      Auto-detect camera SD card and ACASIS archival storage
  tvas --watch              Start watching for SD cards
  tvas --volume /Volumes/DJI_POCKET3 --project "Tokyo Day 1"
  tvas --archival-path /Volumes/MySSD --proxy-path ~/Videos
  tvas --analysis .         Analyze video files in the current directory
        """,
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch for SD card insertions",
    )

    parser.add_argument(
        "--volume",
        type=Path,
        help="Process a specific volume/SD card (auto-detects if not specified)",
    )

    parser.add_argument(
        "--analysis",
        type=Path,
        nargs='?',
        const=Path('.'),
        help="Analyze video files in a directory (defaults to current directory)",
    )

    parser.add_argument(
        "--project",
        type=str,
        help="Project name for organization",
    )

    parser.add_argument(
        "--archival-path",
        type=Path,
        help="Path for archival storage (auto-detects /Volumes/Acasis if not specified)",
    )

    parser.add_argument(
        "--proxy-path",
        type=Path,
        default=Path.home() / "Movies" / "Vlog",
        help="Path for local proxy/timeline storage (default: ~/Movies/Vlog)",
    )

    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable VLM analysis (use OpenCV only)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_VLM_MODEL,
        help="mlx-vlm model for VLM analysis",
    )

    parser.add_argument(
        "--api-base",
        type=str,
        help="Base URL for VLM API (e.g. https://openrouter.ai/api/v1)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for VLM API",
    )

    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Use OpenRouter API (sets default base URL)",
    )

    parser.add_argument(
        "--lmstudio",
        action="store_true",
        help="Use LM Studio local server (sets default base URL)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        help="Preferred provider(s) for OpenRouter (comma-separated)",
    )

    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve all AI decisions (skip UI)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle API configuration shortcuts
    api_base = args.api_base
    if args.openrouter:
        api_base = "https://openrouter.ai/api/v1"
        args.api_key = get_openrouter_api_key(args.api_key)
        if not args.model or args.model == DEFAULT_VLM_MODEL:
            args.model = "qwen/qwen3-vl-235b-a22b-instruct"
    elif args.lmstudio:
        api_base = "http://localhost:1234/v1"

    app = TVASApp(
        archival_path=args.archival_path,
        proxy_path=args.proxy_path,
        use_vlm=not args.no_vlm,
        vlm_model=args.model,
        auto_approve=args.auto_approve,
        api_base=api_base,
        api_key=args.api_key,
        provider_preferences=args.provider,
    )

    if args.analysis:
        if not args.analysis.exists():
            logger.error(f"Directory not found: {args.analysis}")
            sys.exit(1)
        
        results = app.process_directory(args.analysis)
        
        if results["success"]:
            logger.info("Analysis complete!")
            logger.info(f"Files processed: {results['files_processed']}")
            logger.info(f"Clips analyzed: {results['clips_analyzed']}")
        else:
            logger.error("Analysis failed:")
            for error in results["errors"]:
                logger.error(f"  - {error}")
            sys.exit(1)
        sys.exit(0)

    if args.watch:
        app.run_daemon()
    elif args.volume:
        if not args.volume.exists():
            logger.error(f"Volume does not exist: {args.volume}")
            sys.exit(1)

        results = app.process_volume(args.volume, args.project)

        if results["success"]:
            logger.info("Processing complete!")
            logger.info(f"Files processed: {results['files_processed']}")
            logger.info(f"Clips analyzed: {results['clips_analyzed']}")
            logger.info(f"Timeline: {results['timeline_path']}")
        else:
            logger.error("Processing failed:")
            for error in results["errors"]:
                logger.error(f"  - {error}")
            sys.exit(1)
    else:
        # Auto-detect camera volumes or use archival path
        logger.info("No volume specified - searching for camera SD cards...")
        camera_volumes = find_camera_volumes()
        
        if not camera_volumes:
            # No SD card found - check if we can process from archival path
            if app.archival_path and app.archival_path.exists():
                logger.info(f"No SD card found, but archival path exists: {app.archival_path}")
                logger.info("Checking for previously copied files to process...")
                
                # Look for the most recent project directory in archival path
                project_dirs = sorted(
                    [d for d in app.archival_path.iterdir() if d.is_dir() and not d.name.startswith('.')],
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                
                if project_dirs:
                    latest_project = project_dirs[0]
                    logger.info(f"Found recent project directory: {latest_project.name}")
                    
                    # Process from archival path (files already copied)
                    results = app.process_from_archival(latest_project, args.project)
                    
                    if results["success"]:
                        logger.info("Processing complete!")
                        logger.info(f"Files processed: {results['files_processed']}")
                        logger.info(f"Clips analyzed: {results['clips_analyzed']}")
                        logger.info(f"Timeline: {results['timeline_path']}")
                    else:
                        logger.error("Processing failed:")
                        for error in results["errors"]:
                            logger.error(f"  - {error}")
                        sys.exit(1)
                else:
                    logger.error("No project directories found in archival path")
                    logger.error("Please insert an SD card or specify --volume")
                    sys.exit(1)
            else:
                logger.error("No camera volumes found and no archival path available")
                logger.error("Please insert an SD card or specify --volume")
                parser.print_help()
                sys.exit(1)
        elif len(camera_volumes) == 1:
            logger.info(f"Found camera volume: {camera_volumes[0]}")
            results = app.process_volume(camera_volumes[0], args.project)
            
            if results["success"]:
                logger.info("Processing complete!")
                logger.info(f"Files processed: {results['files_processed']}")
                logger.info(f"Clips analyzed: {results['clips_analyzed']}")
                logger.info(f"Timeline: {results['timeline_path']}")
            else:
                logger.error("Processing failed:")
                for error in results["errors"]:
                    logger.error(f"  - {error}")
                sys.exit(1)
        else:
            logger.error(f"Multiple camera volumes found: {[str(v) for v in camera_volumes]}")
            logger.error("Please specify which volume to process with --volume")
            sys.exit(1)


if __name__ == "__main__":
    main()

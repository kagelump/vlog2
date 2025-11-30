"""Main entry point for TVAS (Travel Vlog Automation System).

This module orchestrates the complete workflow from SD card detection
to timeline generation.
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tvas.analysis import ClipDecision, analyze_clips_batch
from tvas.ingestion import CameraType, detect_camera_type, ingest_volume
from tvas.proxy import ProxyConfig, ProxyType, generate_proxies_batch
from tvas.review_ui import (
    ClipReviewItem,
    UserDecision,
    check_toga_available,
    create_review_items_from_analysis,
    run_review_ui,
)
from tvas.timeline import TimelineConfig, create_timeline_from_analysis, export_analysis_json
from tvas.watcher import VolumeWatcher, check_watchdog_available, is_camera_volume

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
        vlog_base_path: Path | None = None,
        use_vlm: bool = True,
        vlm_model: str = "qwen2.5-vl:7b",
        auto_approve: bool = False,
    ):
        """Initialize the TVAS application.

        Args:
            vlog_base_path: Base path for vlog storage (default: ~/Movies/Vlog).
            use_vlm: Whether to use VLM for analysis.
            vlm_model: Ollama model name for VLM.
            auto_approve: Auto-approve all AI decisions without UI.
        """
        self.vlog_base_path = vlog_base_path or Path.home() / "Movies" / "Vlog"
        self.use_vlm = use_vlm
        self.vlm_model = vlm_model
        self.auto_approve = auto_approve
        self.watcher: VolumeWatcher | None = None
        self._running = False

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

        # Stage 1: Ingest files
        logger.info("Stage 1: Ingesting files...")
        try:
            session = ingest_volume(
                volume_path,
                self.vlog_base_path,
                project_name,
                progress_callback=lambda name, i, total: logger.info(
                    f"Copying {i}/{total}: {name}"
                ),
            )
            results["files_processed"] = len(session.files)
            logger.info(f"Ingested {len(session.files)} files")
        except Exception as e:
            results["errors"].append(f"Ingestion failed: {e}")
            logger.error(f"Ingestion failed: {e}")
            return results

        if not session.files:
            results["errors"].append("No files found to process")
            return results

        # Stage 2: Generate AI proxies
        logger.info("Stage 2: Generating AI proxies...")
        source_files = [f.destination_path for f in session.files if f.destination_path]

        # Determine cache directory
        cache_dir = self.vlog_base_path / f"{datetime.now().strftime('%Y-%m-%d')}_{project_name}" / ".cache"

        try:
            proxy_results = generate_proxies_batch(
                source_files,
                cache_dir,
                ProxyType.AI_PROXY,
                ProxyConfig(use_hardware_accel=True),
            )
            successful_proxies = [r for r in proxy_results if r.success]
            logger.info(f"Generated {len(successful_proxies)}/{len(proxy_results)} proxies")
        except Exception as e:
            results["errors"].append(f"Proxy generation failed: {e}")
            logger.error(f"Proxy generation failed: {e}")
            # Continue without proxies - will analyze originals
            proxy_results = []

        # Create clips list for analysis
        clips_to_analyze = []
        for source_file in source_files:
            # Find corresponding proxy
            proxy_path = None
            for pr in proxy_results:
                if pr.success and pr.source_path == source_file:
                    proxy_path = pr.proxy_path
                    break
            clips_to_analyze.append((source_file, proxy_path))

        # Stage 3: AI Analysis
        logger.info("Stage 3: Analyzing clips...")
        try:
            analyses = analyze_clips_batch(
                clips_to_analyze,
                use_vlm=self.use_vlm,
                model_name=self.vlm_model,
            )
            results["clips_analyzed"] = len(analyses)

            # Export analysis for debugging
            analysis_json = cache_dir / "analysis.json"
            export_analysis_json(analyses, analysis_json)
            logger.info(f"Analysis exported to {analysis_json}")
        except Exception as e:
            results["errors"].append(f"Analysis failed: {e}")
            logger.error(f"Analysis failed: {e}")
            return results

        # Stage 4: User Review
        reviewed_clips = None
        if not self.auto_approve:
            logger.info("Stage 4: Opening review UI...")
            if check_toga_available():
                try:
                    review_items = create_review_items_from_analysis(analyses)
                    reviewed_clips = run_review_ui(review_items)
                except Exception as e:
                    logger.warning(f"UI failed, using auto-approve: {e}")
                    self.auto_approve = True
            else:
                logger.info("Toga not available - auto-approving AI decisions")
                self.auto_approve = True

        if self.auto_approve:
            # Create review items with auto-approve
            review_items = create_review_items_from_analysis(analyses)
            for item in review_items:
                item.user_decision = UserDecision.APPROVE
            reviewed_clips = review_items

        # Stage 5: Generate Timeline
        logger.info("Stage 5: Generating timeline...")
        timeline_path = (
            self.vlog_base_path
            / f"{datetime.now().strftime('%Y-%m-%d')}_{project_name}"
            / f"{project_name}_timeline.otio"
        )

        try:
            result_path = create_timeline_from_analysis(
                analyses,
                reviewed_clips,
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
  tvas --watch              Start watching for SD cards
  tvas --volume /Volumes/DJI_POCKET3 --project "Tokyo Day 1"
  tvas --volume /Volumes/SONY_A7C --auto-approve
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
        help="Process a specific volume/SD card",
    )

    parser.add_argument(
        "--project",
        type=str,
        help="Project name for organization",
    )

    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path.home() / "Movies" / "Vlog",
        help="Base path for vlog storage (default: ~/Movies/Vlog)",
    )

    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable VLM analysis (use OpenCV only)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-vl:7b",
        help="Ollama model for VLM analysis",
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

    app = TVASApp(
        vlog_base_path=args.base_path,
        use_vlm=not args.no_vlm,
        vlm_model=args.model,
        auto_approve=args.auto_approve,
    )

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
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()

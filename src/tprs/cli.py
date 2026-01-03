"""Command-line interface for Travel Photo Rating System (TPRS).

Scans SD cards for JPEG photos and generates XMP sidecar files with
AI-powered ratings, keywords, and descriptions.
"""

import argparse
import logging
import multiprocessing
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from shared import DEFAULT_VLM_MODEL, get_openrouter_api_key
from tprs.tprs import (
    find_jpeg_photos,
    process_photos_batch,
)
from shared.vlm_client import check_lmstudio_running

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)





def main():
    """Main entry point for TPRS CLI."""
    parser = argparse.ArgumentParser(
        description="Travel Photo Rating System (TPRS) - Generate XMP sidecars for photos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tprs                                            # Launch GUI for folder selection
  tprs /Volumes/SD_CARD                           # Launch GUI with pre-selected folder
  tprs /Volumes/SD_CARD --headless                # Scan SD card in headless mode
  tprs /path/to/photos --headless --output /tmp/xmp # Headless with custom output directory
  tprs /path/to/photos --headless --model qwen2-vl-7b # Headless with specific model
        """,
    )

    parser.add_argument(
        "directory",
        type=Path,
        nargs='?',
        help="Directory to scan for JPEG photos (optional in GUI mode, required in headless mode)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory for XMP files (default: same as photo location)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_VLM_MODEL,
        help=f"mlx-vlm model for photo analysis (default: {DEFAULT_VLM_MODEL})",
    )

    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=1,
        help="Number of concurrent processes/threads for burst analysis (default: 1)",
    )

    parser.add_argument(
        "--api-base",
        type=str,
        help="Base URL for custom API endpoint (e.g., http://localhost:1234/v1). If set, local MLX model is skipped.",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default="lm-studio",
        help="API key for custom endpoint (default: lm-studio)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find photos but don't process them (headless mode only)",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without GUI (default is GUI mode)",
    )

    parser.add_argument(
        "--lmstudio",
        action="store_true",
        help="Use LM Studio local server (sets api-base and model)",
    )

    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Use OpenRouter API (sets api-base and loads key from ~/.openrouterkey if not provided)",
    )

    parser.add_argument(
        "--openrouter-provider",
        type=str,
        help="Comma-separated list of providers to prefer for OpenRouter (e.g. 'Alibaba,DeepInfra'). Defaults to 'Alibaba' if model is default.",
    )

    # Filter out macOS process serial number argument
    if sys.platform == 'darwin':
        sys.argv = [arg for arg in sys.argv if not arg.startswith('-psn_')]

    args = parser.parse_args()

    # Auto-detect LM Studio
    if not args.lmstudio and not args.api_base and not args.openrouter:
        if check_lmstudio_running():
            logger.info("Auto-detected LM Studio at http://localhost:1234. Enabling --lmstudio mode.")
            args.lmstudio = True

    if args.lmstudio:
        if args.api_base is None:
            args.api_base = "http://localhost:1234/v1"
        if args.model == DEFAULT_VLM_MODEL:
            args.model = "qwen/qwen3-vl"

    if args.openrouter:
        if args.api_base is None:
            args.api_base = "https://openrouter.ai/api/v1"
        
        if args.model == DEFAULT_VLM_MODEL:
            args.model = "qwen/qwen3-vl-8b-instruct"
            if not args.openrouter_provider:
                args.openrouter_provider = "Alibaba"
        
        if args.api_key == "lm-studio":
            args.api_key = get_openrouter_api_key(None)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate flag combinations
    if args.dry_run and not args.headless:
        logger.error("--dry-run flag requires --headless mode")
        parser.print_help()
        sys.exit(1)

    # Default to GUI mode unless --headless is specified
    if not args.headless:
        from tprs.ui import main as ui_main
        app = ui_main(args.directory, args.output, args.model, args.api_base, args.api_key)
        app.main_loop()
        sys.exit(0)

    # Check directory exists (required for headless mode)
    if not args.directory:
        logger.error("Directory argument is required for headless mode")
        parser.print_help()
        sys.exit(1)
    
    if not args.directory.exists():
        logger.error(f"Directory does not exist: {args.directory}")
        sys.exit(1)

    # Find photos
    logger.info(f"Scanning for JPEG photos in {args.directory}")
    photos = find_jpeg_photos(args.directory)

    if not photos:
        logger.warning("No JPEG photos found")
        sys.exit(0)

    logger.info(f"Found {len(photos)} JPEG photos")

    if args.dry_run:
        logger.info("Dry run - photos found:")
        for photo in photos:
            logger.info(f"  {photo}")
        sys.exit(0)

    # Create output directory if specified
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        logger.info(f"XMP files will be saved to: {args.output}")

    # Process photos
    logger.info("Starting photo analysis...")
    try:
        results = process_photos_batch(
            photos, 
            args.model, 
            args.output,
            api_base=args.api_base,
            api_key=args.api_key,
            num_workers=args.processes,
            provider_preferences=args.openrouter_provider
        )

        logger.info("=" * 60)
        logger.info("Processing complete!")
        logger.info(f"Processed {len(results)} photos")
        logger.info(f"XMP sidecar files generated")

        # Summary
        logger.info("\nSummary:")
        
        total_photos = len(results)
        rating_counts = {i: 0 for i in range(1, 6)}
        keyword_counts = {}

        for analysis, _ in results:
            # Count ratings
            r = analysis.rating
            if r in rating_counts:
                rating_counts[r] += 1
            
            # Count keywords
            for k in analysis.keywords:
                k_lower = k.lower()
                keyword_counts[k_lower] = keyword_counts.get(k_lower, 0) + 1

        logger.info(f"Total Photos Analyzed: {total_photos}")
        
        logger.info("\nRating Distribution:")
        for rating in range(1, 6):
            count = rating_counts.get(rating, 0)
            percentage = (count / total_photos * 100) if total_photos > 0 else 0
            logger.info(f"  {rating} Stars: {count} ({percentage:.1f}%)")

        logger.info("\nTop 10 Keywords:")
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for keyword, count in sorted_keywords:
            logger.info(f"  {keyword}: {count}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

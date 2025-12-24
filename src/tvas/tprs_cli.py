"""Command-line interface for Travel Photo Rating System (TPRS).

Scans SD cards for JPEG photos and generates XMP sidecar files with
AI-powered ratings, keywords, and descriptions.
"""

import argparse
import logging
import sys
from pathlib import Path

from tvas.tprs import (
    DEFAULT_VLM_MODEL,
    find_jpeg_photos,
    process_photos_batch,
)

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
  tprs /Volumes/SD_CARD                    # Scan SD card and generate XMP files
  tprs /path/to/photos --output /tmp/xmp   # Output XMP files to specific directory
  tprs /path/to/photos --model qwen2-vl-7b # Use specific model
        """,
    )

    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to scan for JPEG photos (e.g., SD card mount point)",
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
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find photos but don't process them",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check directory exists
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
        results = process_photos_batch(photos, args.model, args.output)

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
    main()

"""Travel Photo Rating System (TPRS)

This module provides functionality to scan SD cards for JPEG photos,
analyze them using Qwen3 VL, and generate XMP sidecar files with ratings,
keywords, and descriptions for use with DxO PhotoLab and other tools.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from PIL import Image

logger = logging.getLogger(__name__)

# mlx-vlm is required
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Default model for mlx-vlm
DEFAULT_VLM_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-8bit"


@dataclass
class PhotoAnalysis:
    """Analysis result for a photo."""

    photo_path: Path
    rating: int  # 1-5 stars
    keywords: list[str]  # 5 keywords
    description: str  # Caption
    raw_response: Optional[str] = None


def find_jpeg_photos(directory: Path) -> list[Path]:
    """Find all JPEG photos in a directory and subdirectories.

    Args:
        directory: Directory to search for JPEG files.

    Returns:
        List of paths to JPEG files.
    """
    jpeg_extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    photos = []

    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return photos

    for ext in jpeg_extensions:
        photos.extend(directory.rglob(f"*{ext}"))

    logger.info(f"Found {len(photos)} JPEG photos in {directory}")
    return sorted(photos)


def resize_image(image_path: Path, max_dimension: int = 1024) -> Optional[Path]:
    """Resize image if it exceeds max_dimension. Returns temp path or None if no resize needed."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width <= max_dimension and height <= max_dimension:
                return None
            
            # Calculate new size
            ratio = min(max_dimension / width, max_dimension / height)
            new_size = (int(width * ratio), int(height * ratio))
                
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save to temp file
            tf = tempfile.NamedTemporaryFile(suffix=image_path.suffix, delete=False)
            img.save(tf.name)
            tf.close()
            return Path(tf.name)
    except Exception as e:
        logger.warning(f"Failed to resize image {image_path}: {e}")
        return None


def analyze_photo_vlm(
    photo_path: Path,
    model,
    processor,
    config,
) -> Optional[PhotoAnalysis]:
    """Analyze a photo using Vision Language Model via mlx-vlm.

    Args:
        photo_path: Path to the photo file.
        model: Loaded mlx-vlm model.
        processor: Loaded processor.
        config: Loaded config.

    Returns:
        PhotoAnalysis with rating, keywords, and description, or None if analysis fails.
    """
    # Resize image if needed
    temp_path = resize_image(photo_path)
    image_path_str = str(temp_path) if temp_path else str(photo_path)

    try:
        # First prompt: Get star rating
        rating_prompt = """Analyze this photograph for technical quality, sharpness, and composition. Give it a star rating from 1 to 5. Output only the integer."""

        rating_response = ""
        try:
            logger.debug(f"Analyzing rating for {photo_path.name}")
            formatted_prompt = apply_chat_template(
                processor, config, rating_prompt, num_images=1
            )

            rating_response = generate(
                model,
                processor,
                formatted_prompt,
                [image_path_str],
                verbose=False,
                max_tokens=10,
            )

            # Extract rating (should be just a number)
            rating_str = rating_response.strip()
            # Try to extract first digit
            rating = 3  # default
            for char in rating_str:
                if char.isdigit():
                    rating = int(char)
                    if 1 <= rating <= 5:
                        break
            if not (1 <= rating <= 5):
                rating = 3

            logger.debug(f"Rating: {rating}")
        except Exception as e:
            logger.error(f"Rating analysis failed: {e}")
            rating = 3

        # Second prompt: Get keywords
        keywords_prompt = """List 5 keywords describing the image content."""

        keywords_response = ""
        try:
            logger.debug(f"Analyzing keywords for {photo_path.name}")
            formatted_prompt = apply_chat_template(
                processor, config, keywords_prompt, num_images=1
            )

            keywords_response = generate(
                model,
                processor,
                formatted_prompt,
                [image_path_str],
                verbose=False,
                max_tokens=100,
            )

            # Parse keywords from response
            # Response might be: "sunset, beach, ocean, waves, sky" or "1. sunset\n2. beach..."
            keywords = []
            # Remove common numbering/bullets
            for sep in ["\n", ",", ";", "|"]:
                if sep in keywords_response:
                    parts = keywords_response.split(sep)
                    for part in parts:
                        # Clean up the part
                        clean = part.strip()
                        # Remove leading numbers/bullets
                        clean = clean.lstrip("0123456789.- ")
                        if clean and len(clean) > 1:
                            keywords.append(clean)
                        if len(keywords) >= 5:
                            break
                    break

            # If we didn't parse any keywords, split by spaces
            if not keywords:
                words = keywords_response.split()[:5]
                keywords = [w.strip(".,;:") for w in words if len(w.strip(".,;:")) > 1]

            # Ensure we have exactly 5 keywords
            while len(keywords) < 5:
                keywords.append("general")
            keywords = keywords[:5]

            logger.debug(f"Keywords: {keywords}")
        except Exception as e:
            logger.error(f"Keyword analysis failed: {e}")
            keywords = ["photo", "image", "travel", "memory", "capture"]

        # Third prompt: Get description/caption
        description_prompt = """Write a brief caption for this photo in one sentence, describing what you see."""

        description_response = ""
        try:
            logger.debug(f"Analyzing description for {photo_path.name}")
            formatted_prompt = apply_chat_template(
                processor, config, description_prompt, num_images=1
            )

            description_response = generate(
                model,
                processor,
                formatted_prompt,
                [image_path_str],
                verbose=False,
                max_tokens=150,
            )

            description = description_response.strip()
            # Limit length
            if len(description) > 300:
                description = description[:297] + "..."

            logger.debug(f"Description: {description[:50]}...")
        except Exception as e:
            logger.error(f"Description analysis failed: {e}")
            description = "Photo from travel collection."

        return PhotoAnalysis(
            photo_path=photo_path,
            rating=rating,
            keywords=keywords,
            description=description,
            raw_response=f"Rating: {rating_response}\nKeywords: {keywords_response}\nDescription: {description_response}",
        )
    finally:
        if temp_path and temp_path.exists():
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")


def generate_xmp_sidecar(analysis: PhotoAnalysis, output_path: Optional[Path] = None) -> Path:
    """Generate XMP sidecar file for a photo analysis.

    Creates an XMP file compatible with DxO PhotoLab and other tools.

    Args:
        analysis: PhotoAnalysis result.
        output_path: Optional output path. If None, uses photo_name.xmp.

    Returns:
        Path to the generated XMP file.
    """
    if output_path is None:
        # Generate sidecar name: image001.jpg -> image001.xmp
        output_path = analysis.photo_path.with_suffix(".xmp")

    # Create XMP structure
    # Using proper XMP namespaces
    xmpmeta = ET.Element("x:xmpmeta")
    xmpmeta.set("xmlns:x", "adobe:ns:meta/")

    rdf = ET.SubElement(xmpmeta, "rdf:RDF")
    rdf.set("xmlns:rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")

    description = ET.SubElement(rdf, "rdf:Description")
    description.set("rdf:about", "")
    description.set("xmlns:xmp", "http://ns.adobe.com/xap/1.0/")
    description.set("xmlns:dc", "http://purl.org/dc/elements/1.1/")

    # Add rating
    rating_elem = ET.SubElement(description, "xmp:Rating")
    rating_elem.text = str(analysis.rating)

    # Add keywords (dc:subject is a bag of strings)
    subject = ET.SubElement(description, "dc:subject")
    bag = ET.SubElement(subject, "rdf:Bag")
    for keyword in analysis.keywords:
        li = ET.SubElement(bag, "rdf:li")
        li.text = keyword

    # Add description/caption (dc:description is an alt text)
    desc_elem = ET.SubElement(description, "dc:description")
    alt = ET.SubElement(desc_elem, "rdf:Alt")
    li = ET.SubElement(alt, "rdf:li")
    li.set("xml:lang", "x-default")
    li.text = analysis.description

    # Write to file with proper XML formatting
    tree = ET.ElementTree(xmpmeta)
    ET.indent(tree, space="  ")

    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="UTF-8", xml_declaration=False)
        f.write(b"\n")

    logger.info(f"Generated XMP sidecar: {output_path}")
    return output_path


def process_photos_batch(
    photos: list[Path],
    model_name: str = DEFAULT_VLM_MODEL,
    output_dir: Optional[Path] = None,
) -> list[tuple[PhotoAnalysis, Path]]:
    """Process a batch of photos and generate XMP sidecars.

    Args:
        photos: List of photo paths to process.
        model_name: mlx-vlm model name.
        output_dir: Optional output directory for XMP files.

    Returns:
        List of (PhotoAnalysis, xmp_path) tuples.
    """
    results = []

    # Load model once
    try:
        logger.info(f"Loading mlx-vlm model: {model_name}")
        model, processor = load(model_name)
        config = load_config(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return results

    for i, photo_path in enumerate(photos):
        logger.info(f"Processing photo {i + 1}/{len(photos)}: {photo_path.name}")

        # Analyze photo
        analysis = analyze_photo_vlm(photo_path, model, processor, config)

        if analysis is None:
            logger.warning(f"Skipping XMP generation for {photo_path.name} due to analysis failure")
            continue

        # Generate XMP
        if output_dir:
            xmp_path = output_dir / f"{photo_path.stem}.xmp"
        else:
            xmp_path = None

        xmp_file = generate_xmp_sidecar(analysis, xmp_path)

        results.append((analysis, xmp_file))

    logger.info(f"Processed {len(results)} photos")
    return results

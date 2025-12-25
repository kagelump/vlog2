"""Travel Photo Rating System (TPRS)

This module provides functionality to scan SD cards for JPEG photos,
analyze them using Qwen3 VL, and generate XMP sidecar files with ratings,
keywords, and descriptions for use with DxO PhotoLab and other tools.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator, Callable, Any
from xml.etree import ElementTree as ET

from PIL import Image, ExifTags

from tvas import load_prompt

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
    primary_subject: Optional[str] = None
    primary_subject_bounding_box: Optional[list[int]] = None
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


def get_capture_time(image_path: Path) -> datetime:
    """Get capture time from EXIF data."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return datetime.fromtimestamp(image_path.stat().st_mtime)
            
            # 36867 is DateTimeOriginal, 306 is DateTime
            date_str = exif.get(36867) or exif.get(306)
            
            if date_str:
                try:
                    return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass
                    
            return datetime.fromtimestamp(image_path.stat().st_mtime)
    except Exception:
        return datetime.fromtimestamp(image_path.stat().st_mtime)


def are_photos_in_same_burst(
    photo1: Path,
    photo2: Path,
    model,
    processor,
    config
) -> bool:
    """Use VLM to determine if two photos belong to the same burst."""
    p1_resized = None
    p2_resized = None
    try:
        # Resize for speed
        p1_resized = resize_image(photo1, max_dimension=512)
        p2_resized = resize_image(photo2, max_dimension=512)
        
        image_paths = [
            str(p1_resized if p1_resized else photo1),
            str(p2_resized if p2_resized else photo2)
        ]
        
        prompt = load_prompt("burst_similarity.txt")

        formatted_prompt = apply_chat_template(
            processor, config, prompt, num_images=2
        )
        
        response = generate(
            model,
            processor,
            formatted_prompt,
            image_paths,
            verbose=False,
            max_tokens=50,
        )
        
        if hasattr(response, "text"):
            text = response.text
        else:
            text = str(response)
            
        # Clean JSON
        clean_text = text.strip()
        if clean_text.startswith("```json"): clean_text = clean_text[7:]
        if clean_text.startswith("```"): clean_text = clean_text[3:]
        if clean_text.endswith("```"): clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        
        data = json.loads(clean_text)
        return bool(data.get("same_burst", False))
        
    except Exception as e:
        logger.warning(f"Burst comparison failed for {photo1.name} and {photo2.name}: {e}")
        return False
    finally:
        # Cleanup temp files
        if p1_resized and p1_resized.exists():
            try:
                os.unlink(p1_resized)
            except Exception:
                pass
        if p2_resized and p2_resized.exists():
            try:
                os.unlink(p2_resized)
            except Exception:
                pass


def generate_bursts(
    photos: list[Path], 
    model,
    processor,
    config,
    threshold_minutes: float = 5.0,
    comparison_callback: Optional[Callable[[Path, Path], None]] = None
) -> Iterator[list[Path]]:
    """Yield bursts of photos based on capture time and visual similarity."""
    if not photos:
        return
        
    # Sort by capture time
    photos_with_time = []
    for p in photos:
        photos_with_time.append((p, get_capture_time(p)))
    
    photos_with_time.sort(key=lambda x: x[1])
    
    current_burst = [photos_with_time[0][0]]
    prev_time = photos_with_time[0][1]
    prev_photo = photos_with_time[0][0]
    
    logger.info("Starting burst detection...")
    
    for i in range(1, len(photos_with_time)):
        curr_photo, curr_time = photos_with_time[i]
        
        time_diff = (curr_time - prev_time).total_seconds() / 60.0
        
        is_same_burst = False
        if time_diff > threshold_minutes:
            # Definitely different burst
            is_same_burst = False
        else:
            # Check with model
            logger.info(f"Checking burst similarity: {prev_photo.name} vs {curr_photo.name} (diff: {time_diff:.1f}m)")
            if comparison_callback:
                comparison_callback(prev_photo, curr_photo)
            is_same_burst = are_photos_in_same_burst(prev_photo, curr_photo, model, processor, config)
            
        if is_same_burst:
            current_burst.append(curr_photo)
        else:
            yield current_burst
            current_burst = [curr_photo]
            
        prev_time = curr_time
        prev_photo = curr_photo
            
    if current_burst:
        yield current_burst


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


def crop_image(image_path: Path, bbox: list[int]) -> Optional[Path]:
    """Crop image to bounding box. bbox is [ymin, xmin, ymax, xmax] on 0-1000 scale."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            ymin, xmin, ymax, xmax = bbox
            
            # Convert 0-1000 scale to pixels
            left = int((xmin / 1000) * width)
            top = int((ymin / 1000) * height)
            right = int((xmax / 1000) * width)
            bottom = int((ymax / 1000) * height)
            
            # Ensure valid crop
            if left >= right or top >= bottom:
                return None
                
            cropped = img.crop((left, top, right, bottom))
            
            # Save to temp file
            tf = tempfile.NamedTemporaryFile(suffix=image_path.suffix, delete=False)
            cropped.save(tf.name)
            tf.close()
            return Path(tf.name)
    except Exception as e:
        logger.warning(f"Failed to crop image {image_path}: {e}")
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
        # Single prompt for JSON output
        json_prompt = load_prompt("photo_analysis.txt")

        logger.debug(f"Analyzing {photo_path.name} with JSON prompt")
        formatted_prompt = apply_chat_template(
            processor, config, json_prompt, num_images=1
        )

        response = generate(
            model,
            processor,
            formatted_prompt,
            [image_path_str],
            verbose=False,
            max_tokens=500,
        )
        
        # Handle GenerationResult object
        if hasattr(response, "text"):
            response_text = response.text
        else:
            response_text = str(response)

        # Clean up response text (remove markdown code blocks if present)
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        try:
            data = json.loads(clean_text)
            
            rating_str = str(data.get("rating", "OK")).upper()
            rating_map = {
                "UNUSABLE": 1,
                "BAD": 2,
                "OK": 3,
                "GOOD": 4,
                "EXCELLENT": 5
            }
            rating = rating_map.get(rating_str, 3)
            
            primary_subject = data.get("primary_subject")
            primary_subject_bounding_box = data.get("primary_subject_bounding_box")

            keywords = data.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = str(keywords).split(",")
            
            # Ensure 5 keywords
            keywords = [str(k).strip() for k in keywords if str(k).strip()]
            
            # Prepend primary_subject if available
            if primary_subject:
                ps_clean = str(primary_subject).strip()
                if ps_clean:
                    if ps_clean in keywords:
                        keywords.remove(ps_clean)
                    keywords.insert(0, ps_clean)

            while len(keywords) < 5:
                keywords.append("general")
            keywords = keywords[:5]
            
            description = str(data.get("description", "Photo from travel collection."))
            if len(description) > 300:
                description = description[:297] + "..."
            
            # Secondary analysis: Check subject sharpness
            if primary_subject and primary_subject_bounding_box and isinstance(primary_subject_bounding_box, list) and len(primary_subject_bounding_box) == 4:
                logger.debug(f"Performing secondary subject analysis for {primary_subject}")
                crop_path = crop_image(photo_path, primary_subject_bounding_box)
                
                if crop_path:
                    # Resize crop if needed to avoid memory issues
                    resized_crop = resize_image(crop_path)
                    final_crop_path = resized_crop if resized_crop else crop_path
                    
                    try:
                        subject_prompt_template = load_prompt("subject_sharpness.txt")
                        subject_prompt = subject_prompt_template.format(primary_subject=primary_subject)
                        
                        formatted_subject_prompt = apply_chat_template(
                            processor, config, subject_prompt, num_images=1
                        )
                        
                        subject_response = generate(
                            model,
                            processor,
                            formatted_subject_prompt,
                            [str(final_crop_path)],
                            verbose=False,
                            max_tokens=50,
                        )
                        
                        if hasattr(subject_response, "text"):
                            subject_text = subject_response.text
                        else:
                            subject_text = str(subject_response)
                            
                        # Clean JSON
                        clean_subject = subject_text.strip()
                        if clean_subject.startswith("```json"): clean_subject = clean_subject[7:]
                        if clean_subject.startswith("```"): clean_subject = clean_subject[3:]
                        if clean_subject.endswith("```"): clean_subject = clean_subject[:-3]
                        clean_subject = clean_subject.strip()
                        
                        try:
                            subject_data = json.loads(clean_subject)
                            blur_level = subject_data.get("blur_level", "SHARP")
                            
                            if blur_level == "VERY_BLURRY":
                                logger.info(f"Subject '{primary_subject}' detected as VERY_BLURRY. Downgrading rating to 1.")
                                rating = 1
                            elif blur_level == "MINOR_BLURRY":
                                logger.info(f"Subject '{primary_subject}' detected as MINOR_BLURRY. Reducing rating by 1.")
                                rating = max(2, rating - 1)
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse subject analysis response: {subject_text}")
                            
                    except Exception as e:
                        logger.warning(f"Secondary analysis failed: {e}")
                    finally:
                        # Cleanup
                        if resized_crop and resized_crop.exists():
                            try:
                                os.unlink(resized_crop)
                            except Exception:
                                pass
                        if crop_path.exists():
                            try:
                                os.unlink(crop_path)
                            except Exception:
                                pass
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}. Response: {response_text}")
            # Fallback
            rating = 3
            keywords = ["photo", "image", "travel", "memory", "capture"]
            description = "Photo from travel collection."
            primary_subject = None
            primary_subject_bounding_box = None
            
        return PhotoAnalysis(
            photo_path=photo_path,
            rating=rating,
            keywords=keywords,
            description=description,
            primary_subject=primary_subject,
            primary_subject_bounding_box=primary_subject_bounding_box,
            raw_response=response_text,
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return None
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

    logger.info(f"Generated XMP sidecar: {output_path} | Rating: {analysis.rating} | Subjects: {analysis.keywords}")
    return output_path


def get_xmp_info(xmp_path: Path) -> tuple[str, list[str]]:
    """Extract rating and keywords from XMP file."""
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        
        namespaces = {
            'xmp': 'http://ns.adobe.com/xap/1.0/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        }
        
        rating = "Unknown"
        keywords = []
        
        # Search for Rating
        rating_elem = root.find(".//xmp:Rating", namespaces)
        if rating_elem is not None:
            rating = rating_elem.text
            
        # Search for Keywords
        bag = root.find(".//dc:subject/rdf:Bag", namespaces)
        if bag is not None:
            for li in bag.findall("rdf:li", namespaces):
                if li.text:
                    keywords.append(li.text)
                    
        return rating, keywords
    except Exception as e:
        logger.warning(f"Failed to parse XMP {xmp_path}: {e}")
        return "Error", []


def select_best_in_burst(
    burst_analyses: list[PhotoAnalysis],
    model,
    processor,
    config
) -> PhotoAnalysis:
    """Select the best photo from a burst using VLM comparison."""
    # Filter for candidates (rating >= 3)
    candidates = [a for a in burst_analyses if a.rating >= 3]
    
    if not candidates:
        # If all are bad, just return the one with highest rating
        return max(burst_analyses, key=lambda x: x.rating)
        
    if len(candidates) == 1:
        return candidates[0]
        
    # If too many candidates, take top 4 by rating
    candidates.sort(key=lambda x: x.rating, reverse=True)
    candidates = candidates[:4]
    
    # Prepare prompt for comparison
    image_paths = []
    temp_files = []
    
    try:
        for c in candidates:
            # Resize for comparison to save memory and tokens
            resized = resize_image(c.photo_path, max_dimension=512)
            if resized:
                image_paths.append(str(resized))
                temp_files.append(resized)
            else:
                image_paths.append(str(c.photo_path))
        
        prompt = load_prompt("best_in_burst.txt")

        formatted_prompt = apply_chat_template(
            processor, config, prompt, num_images=len(image_paths)
        )
        
        response = generate(
            model,
            processor,
            formatted_prompt,
            image_paths,
            verbose=False,
            max_tokens=100,
        )
        
        if hasattr(response, "text"):
            response_text = response.text
        else:
            response_text = str(response)
            
        # Clean JSON
        clean_text = response_text.strip()
        if clean_text.startswith("```json"): clean_text = clean_text[7:]
        if clean_text.startswith("```"): clean_text = clean_text[3:]
        if clean_text.endswith("```"): clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        
        try:
            data = json.loads(clean_text)
            best_index = int(data.get("best_index", 0))
            if 0 <= best_index < len(candidates):
                return candidates[best_index]
        except Exception as e:
            logger.warning(f"Failed to parse burst selection response: {e}")
            
    except Exception as e:
        logger.error(f"Burst selection failed: {e}")
    finally:
        for tf in temp_files:
            if tf.exists():
                try:
                    os.unlink(tf)
                except Exception:
                    pass
                    
    # Fallback to first candidate (highest rated)
    return candidates[0]


def process_photos_batch(
    photos: list[Path],
    model_name: str = DEFAULT_VLM_MODEL,
    output_dir: Optional[Path] = None,
    status_callback: Optional[Callable[[int, int, Optional[Path], Optional[PhotoAnalysis], Optional[Path]], None]] = None,
) -> list[tuple[PhotoAnalysis, Path]]:
    """Process a batch of photos and generate XMP sidecars.

    Args:
        photos: List of photo paths to process.
        model_name: mlx-vlm model name.
        output_dir: Optional output directory for XMP files.
        status_callback: Callback for progress updates (processed, total, current_photo, last_analysis, comparison_photo).

    Returns:
        List of (PhotoAnalysis, xmp_path) tuples.
    """
    results = []
    photos_to_process = []

    # Check for existing sidecars
    for photo_path in photos:
        if output_dir:
            xmp_path = output_dir / f"{photo_path.stem}.xmp"
        else:
            xmp_path = photo_path.with_suffix(".xmp")
            
        if xmp_path.exists():
            rating, keywords = get_xmp_info(xmp_path)
            logger.info(f"Sidecar exists for {photo_path.name}: rating {rating}, keywords {keywords}")
            
            # Reconstruct analysis for summary
            analysis = PhotoAnalysis(
                photo_path=photo_path,
                rating=int(rating) if rating and rating.isdigit() else 0,
                keywords=keywords,
                description="Loaded from XMP",
                raw_response=None
            )
            results.append((analysis, xmp_path))
        else:
            photos_to_process.append(photo_path)

    total_photos = len(photos)
    processed_count = len(results)

    if status_callback:
        try:
            status_callback(processed_count, total_photos, None, None)
        except TypeError:
             status_callback(processed_count, total_photos, None, None, None)


    if not photos_to_process:
        logger.info("All photos have existing sidecars. No processing needed.")
        return results

    # Load model once
    try:
        logger.info(f"Loading mlx-vlm model: {model_name}")
        model, processor = load(model_name)
        config = load_config(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return results

    # Define comparison callback wrapper
    comparison_cb = None
    if status_callback:
        def comparison_cb(prev, curr):
            try:
                status_callback(processed_count, total_photos, curr, None, prev)
            except TypeError:
                # Fallback for callbacks that don't support the extra argument
                status_callback(processed_count, total_photos, curr, None)

    # Group into bursts
    burst_iterator = generate_bursts(photos_to_process, model, processor, config, comparison_callback=comparison_cb)
    
    for i, burst in enumerate(burst_iterator):
        logger.info(f"Processing burst {i + 1} ({len(burst)} photos)")
        
        burst_analyses = []
        for photo_path in burst:
            # Notify start
            if status_callback:
                try:
                    status_callback(processed_count, total_photos, photo_path, None, None)
                except TypeError:
                    status_callback(processed_count, total_photos, photo_path, None)

            # Analyze photo
            analysis = analyze_photo_vlm(photo_path, model, processor, config)
            
            processed_count += 1
            
            if analysis:
                burst_analyses.append(analysis)
                # Notify end with analysis
                if status_callback:
                    try:
                        status_callback(processed_count, total_photos, photo_path, analysis, None)
                    except TypeError:
                        status_callback(processed_count, total_photos, photo_path, analysis)
            else:
                # Notify end (failed)
                if status_callback:
                    try:
                        status_callback(processed_count, total_photos, photo_path, None, None)
                    except TypeError:
                        status_callback(processed_count, total_photos, photo_path, None)

        if not burst_analyses:
            continue

        # Select best in burst if more than 1 photo
        best_photo = None
        if len(burst_analyses) > 1:
            best_photo = select_best_in_burst(burst_analyses, model, processor, config)
            
            # Apply rating constraints
            for analysis in burst_analyses:
                if analysis == best_photo:
                    analysis.keywords.append("BestInBurst")
                else:
                    if analysis.rating >= 5:
                        analysis.rating = 4
                    analysis.keywords.append("BurstDuplicate")
        
        # Generate XMP for all
        for analysis in burst_analyses:
            if output_dir:
                xmp_path = output_dir / f"{analysis.photo_path.stem}.xmp"
            else:
                xmp_path = None

            xmp_file = generate_xmp_sidecar(analysis, xmp_path)
            results.append((analysis, xmp_file))

    logger.info(f"Processed {len(results)} photos")
    return results

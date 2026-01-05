"""Travel Photo Rating System (TPRS)

This module provides functionality to scan SD cards for JPEG photos,
analyze them using Qwen3 VL, and generate XMP sidecar files with ratings,
keywords, and descriptions for use with DxO PhotoLab and other tools.
"""

import json
import logging
import os
import tempfile
import threading
import contextlib
import concurrent.futures
import base64
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator, Callable, Any
from xml.etree import ElementTree as ET

from PIL import Image, ExifTags

from shared import load_prompt, DEFAULT_VLM_MODEL
from shared.vlm_client import VLMClient, VLMResponse

logger = logging.getLogger(__name__)


@dataclass
class PhotoAnalysis:
    """Analysis result for a photo."""

    photo_path: Path
    rating: int  # 1-5 stars
    rating_reason: str
    keywords: list[str]  # 5 keywords
    description: str  # Caption
    primary_subject: Optional[str] = None
    primary_subject_bounding_box: Optional[list[int]] = None
    raw_response: Optional[str] = None
    best_in_burst: bool = False
    blur_level: int = 0  # 0 SHARP 1 MINOR_BLURRY 2 VERY_BLURRY
    burst_id: Optional[str] = None
    provider: Optional[str] = None
    subject_sharpness_check_required: bool = True
    subject_sharpness_check_reason: Optional[str] = None





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





def clean_json_response(text: str) -> str:
    """Clean up response text to extract JSON content."""
    clean_text = text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    return clean_text.strip()


def are_photos_in_same_burst(
    photo1: Path,
    photo2: Path,
    client: VLMClient
) -> tuple[bool, Optional[str]]:
    """Use VLM to determine if two photos belong to the same burst.
    
    Returns:
        Tuple of (is_same_burst, keyword).
    """
    try:
        # Resize for speed using context manager
        with resize_image(photo1, max_dimension=512) as p1_resized, \
             resize_image(photo2, max_dimension=512) as p2_resized:
            
            image_paths = [
                p1_resized if p1_resized else photo1,
                p2_resized if p2_resized else photo2
            ]
            
            prompt = load_prompt("burst_similarity.txt")

            response = client.generate(
                prompt=prompt,
                image_paths=image_paths,
                max_tokens=50
            )
            
            if not response:
                return False, None
            text = response.text
            
            # Clean JSON
            clean_text = clean_json_response(text)

            data = json.loads(clean_text)
            return bool(data.get("same_burst", False)), data.get("keyword")
        
    except Exception as e:
        logger.warning(f"Burst comparison failed for {photo1.name} and {photo2.name}: {e}")
        return False, None


def generate_bursts(
    photos: list[Path], 
    client: VLMClient,
    threshold_minutes: float = 5.0,
    comparison_callback: Optional[Callable[[Path, Path], None]] = None
) -> Iterator[tuple[list[Path], Optional[str]]]:
    """Yield bursts of photos based on capture time and visual similarity.
    
    Returns:
        Iterator of (burst_photos, burst_keyword).
    """
    if not photos:
        return
        
    # Sort by capture time
    photos_with_time = []
    for p in photos:
        photos_with_time.append((p, get_capture_time(p)))
    
    photos_with_time.sort(key=lambda x: x[1])
    
    current_burst = [photos_with_time[0][0]]
    current_keyword = None
    prev_time = photos_with_time[0][1]
    prev_photo = photos_with_time[0][0]
    
    logger.info("Starting burst detection...")
    
    for i in range(1, len(photos_with_time)):
        curr_photo, curr_time = photos_with_time[i]
        
        time_diff = (curr_time - prev_time).total_seconds() / 60.0
        
        is_same_burst = False
        keyword = None
        
        if time_diff > threshold_minutes:
            # Definitely different burst
            is_same_burst = False
        else:
            # Check with model
            logger.info(f"Checking burst similarity: {prev_photo.name} vs {curr_photo.name} (diff: {time_diff:.1f}m)")
            if comparison_callback:
                comparison_callback(prev_photo, curr_photo)
            is_same_burst, keyword = are_photos_in_same_burst(
                prev_photo, curr_photo, client
            )
            
        if is_same_burst:
            current_burst.append(curr_photo)
            if keyword and not current_keyword:
                current_keyword = keyword
        else:
            yield current_burst, current_keyword
            current_burst = [curr_photo]
            current_keyword = None
            
        prev_time = curr_time
        prev_photo = curr_photo
            
    if current_burst:
        yield current_burst, current_keyword


@contextlib.contextmanager
def temporary_image_file(suffix: str = ".jpg") -> Iterator[Path]:
    """Context manager for creating and cleaning up a temporary image file."""
    tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tf.close()
    path = Path(tf.name)
    try:
        yield path
    finally:
        if path.exists():
            try:
                os.unlink(path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {path}: {e}")


@contextlib.contextmanager
def resize_image(image_path: Path, max_dimension: int = 1024) -> Iterator[Optional[Path]]:
    """Resize image if it exceeds max_dimension. Yields temp path or None if no resize needed."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width <= max_dimension and height <= max_dimension:
                yield None
                return
            
            # Calculate new size
            ratio = min(max_dimension / width, max_dimension / height)
            new_size = (int(width * ratio), int(height * ratio))
                
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            with temporary_image_file(suffix=image_path.suffix) as temp_path:
                img.save(temp_path)
                yield temp_path

    except Exception as e:
        logger.warning(f"Failed to resize image {image_path}: {e}")
        yield None


@contextlib.contextmanager
def crop_image(image_path: Path, bbox: list[int]) -> Iterator[Optional[Path]]:
    """Crop image to bounding box. bbox is [xmin, ymin, xmax, ymax] on 0-1000 scale."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            xmin, ymin, xmax, ymax = bbox
            
            # Convert 0-1000 scale to pixels
            left = int((xmin / 1000) * width)
            top = int((ymin / 1000) * height)
            right = int((xmax / 1000) * width)
            bottom = int((ymax / 1000) * height)
            
            # Ensure valid crop
            if left >= right or top >= bottom:
                yield None
                return
                
            cropped = img.crop((left, top, right, bottom))
            
            with temporary_image_file(suffix=image_path.suffix) as temp_path:
                cropped.save(temp_path)
                yield temp_path
    except Exception as e:
        logger.warning(f"Failed to crop image {image_path}: {e}")
        yield None


def expand_bbox(bbox: list[int]) -> list[int]:
  x1, y1, x2, y2 = bbox
  height = y2 - y1

  return [
      max(0, x1 - 100),
      max(y1 - height, 0),
      min(x2 + 100, 1000),
      y2]


def parse_analysis_response(response_text: str, photo_path: Path, provider: Optional[str] = None) -> Optional[PhotoAnalysis]:
    """Parse JSON response from VLM."""
    # Clean up response text (remove markdown code blocks if present)
    clean_text = clean_json_response(response_text)

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
        rating_reason = data.get("rating_reason", "N/A")
        
        primary_subject = data.get("primary_subject")
        primary_subject_bounding_box = None
        if data.get("primary_subject_bounding_box"):
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
            
        subject_sharpness_check_required = data.get("subject_sharpness_check_required", True)
        subject_sharpness_check_reason = data.get("subject_sharpness_check_reason")

        return PhotoAnalysis(
            photo_path=photo_path,
            rating=rating,
            rating_reason=rating_reason,
            keywords=keywords,
            description=description,
            primary_subject=primary_subject,
            primary_subject_bounding_box=primary_subject_bounding_box,
            raw_response=response_text,
            provider=provider,
            subject_sharpness_check_required=subject_sharpness_check_required,
            subject_sharpness_check_reason=subject_sharpness_check_reason
        )
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response for {photo_path}: {e}")
        logger.debug(f"Raw response: {response_text}")
        return None
    except Exception as e:
        logger.error(f"Error processing analysis for {photo_path}: {e}")
        return None


def analyze_photo(
    photo_path: Path,
    client: VLMClient
) -> Optional[PhotoAnalysis]:
    """Analyze a photo using Vision Language Model via mlx-vlm or API.

    Args:
        photo_path: Path to the photo file.
        client: VLMClient instance.

    Returns:
        PhotoAnalysis with rating, keywords, and description, or None if analysis fails.
    """
    try:
        # Resize image if needed
        with resize_image(photo_path) as temp_path:
            image_path = temp_path if temp_path else photo_path
            blur_level_int = 0
            provider = "mlx-vlm"

            # Single prompt for JSON output
            json_prompt = load_prompt("photo_analysis.txt")

            response = client.generate(
                prompt=json_prompt,
                image_paths=[image_path],
                max_tokens=1000
            )
            
            if not response:
                return None
            response_text = response.text
            provider = response.provider or "api"

            analysis = parse_analysis_response(response_text, photo_path, provider=provider)
            if not analysis:
                return None
                
            # Use analysis object directly
            primary_subject = analysis.primary_subject
            primary_subject_bounding_box = analysis.primary_subject_bounding_box
            rating = analysis.rating
            rating_reason = analysis.rating_reason
            keywords = analysis.keywords
            description = analysis.description
            raw_response = analysis.raw_response

            try:
                # Secondary analysis: Check subject sharpness
                if not analysis.subject_sharpness_check_required:
                    logger.info(f"Skipping subject sharpness check for {photo_path.name}: {analysis.subject_sharpness_check_reason}")
                elif primary_subject and primary_subject_bounding_box and isinstance(primary_subject_bounding_box, list) and len(primary_subject_bounding_box) == 4:
                    logger.debug(f"Performing secondary subject analysis for {primary_subject}")
                    
                    with crop_image(photo_path, primary_subject_bounding_box) as crop_path:
                        if crop_path:
                            # Resize crop if needed to avoid memory issues
                            with resize_image(crop_path) as resized_crop:
                                final_crop_path = resized_crop if resized_crop else crop_path
                                
                                try:
                                    subject_prompt_template = load_prompt("subject_sharpness.txt")
                                    subject_prompt = subject_prompt_template.format(primary_subject=primary_subject)
                                    
                                    response = client.generate(
                                        prompt=subject_prompt,
                                        image_paths=[final_crop_path],
                                        max_tokens=100
                                    )
                                    subject_text = response.text if response else None
                                        
                                    if subject_text:
                                        # Clean JSON
                                        clean_subject = clean_json_response(subject_text)
                                        
                                        try:
                                            subject_data = json.loads(clean_subject)
                                            blur_level = subject_data.get("blur_level", "SHARP")
                                            blur_type = subject_data.get("blur_type", "N/A")
                                            
                                            if blur_level == "VERY_BLURRY":
                                                rating_reason += f" BUT Subject '{primary_subject}' detected as VERY_BLURRY ({blur_type}). Downgrading rating to 1."
                                                logger.info(rating_reason)
                                                rating = 1
                                                blur_level_int = 2
                                            elif blur_level == "MINOR_BLURRY":
                                                rating_reason += f" BUT Subject '{primary_subject}' detected as MINOR_BLURRY ({blur_type}). Reducing rating by 1."
                                                logger.info(rating_reason)
                                                rating = max(1, rating - 1)
                                                blur_level_int = 1
                                                
                                        except json.JSONDecodeError:
                                            logger.warning(f"Failed to parse subject analysis response: {subject_text}")
                                        
                                except Exception as e:
                                    logger.warning(f"Secondary analysis failed: {e}")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}. Response: {response_text}")
                # Fallback
                rating = 1
                rating_reason = "N/A"
                keywords = ["failed"]
                description = "Failed analysis."
                primary_subject = None
                primary_subject_bounding_box = None
                
            return PhotoAnalysis(
                photo_path=photo_path,
                rating=rating,
                rating_reason=rating_reason,
                keywords=keywords,
                description=description,
                primary_subject=primary_subject,
                primary_subject_bounding_box=primary_subject_bounding_box,
                raw_response=response_text,
                blur_level=blur_level_int,
                provider=analysis.provider,
            )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return None


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
    description.set("xmlns:tprs", "http://tvas.local/tprs/1.0/")

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

    # Add TPRS specific data
    if analysis.primary_subject:
        tprs_subject = ET.SubElement(description, "tprs:primary_subject")
        tprs_subject.text = analysis.primary_subject
    
    if analysis.rating_reason:
        tprs_rating_reason = ET.SubElement(description, "tprs:rating_reason")
        tprs_rating_reason.text = analysis.rating_reason
    
    if analysis.primary_subject_bounding_box:
        tprs_bbox = ET.SubElement(description, "tprs:primary_subject_bounding_box")
        tprs_bbox.text = json.dumps(analysis.primary_subject_bounding_box)

    if analysis.raw_response:
        tprs_raw = ET.SubElement(description, "tprs:raw_response")
        tprs_raw.text = analysis.raw_response

    if analysis.blur_level is not None:
        tprs_blur = ET.SubElement(description, "tprs:blur_level")
        tprs_blur.text = str(analysis.blur_level)

    if analysis.best_in_burst:
        tprs_best = ET.SubElement(description, "tprs:best_in_burst")
        tprs_best.text = "True"

    if analysis.burst_id:
        tprs_burst = ET.SubElement(description, "tprs:burst_id")
        tprs_burst.text = analysis.burst_id

    if analysis.provider:
        tprs_provider = ET.SubElement(description, "tprs:provider")
        tprs_provider.text = analysis.provider

    # Write to file with proper XML formatting
    tree = ET.ElementTree(xmpmeta)
    ET.indent(tree, space="  ")

    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="UTF-8", xml_declaration=False)
        f.write(b"\n")

    logger.info(f"Generated XMP sidecar: {output_path} | Rating: {analysis.rating} | Subjects: {analysis.keywords}")
    return output_path


def load_analysis_from_xmp(xmp_path: Path, photo_path: Path) -> PhotoAnalysis:
    """Load PhotoAnalysis from XMP file."""
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        
        namespaces = {
            'xmp': 'http://ns.adobe.com/xap/1.0/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'tprs': 'http://tvas.local/tprs/1.0/'
        }
        
        rating = 0
        keywords = []
        description = ""
        primary_subject = None
        rating_reason = "N/A"
        primary_subject_bounding_box = None
        raw_response = None
        
        # Rating
        rating_elem = root.find(".//xmp:Rating", namespaces)
        if rating_elem is not None and rating_elem.text and rating_elem.text.isdigit():
            rating = int(rating_elem.text)
            
        # Keywords
        bag = root.find(".//dc:subject/rdf:Bag", namespaces)
        if bag is not None:
            for li in bag.findall("rdf:li", namespaces):
                if li.text:
                    keywords.append(li.text)
        
        # Description
        desc_elem = root.find(".//dc:description/rdf:Alt/rdf:li", namespaces)
        if desc_elem is not None:
            description = desc_elem.text or ""

        # TPRS Data
        ps_elem = root.find(".//tprs:primary_subject", namespaces)
        if ps_elem is not None:
            primary_subject = ps_elem.text
        
        rating_reason_elem = root.find(".//tprs:rating_reason", namespaces)
        if rating_reason_elem is not None:
            rating_reason = rating_reason_elem.text or "N/A"
            
        bbox_elem = root.find(".//tprs:primary_subject_bounding_box", namespaces)
        if bbox_elem is not None and bbox_elem.text:
            try:
                primary_subject_bounding_box = json.loads(bbox_elem.text)
            except:
                pass
                
        raw_elem = root.find(".//tprs:raw_response", namespaces)
        if raw_elem is not None:
            raw_response = raw_elem.text

        blur_level = 0
        blur_elem = root.find(".//tprs:blur_level", namespaces)
        if blur_elem is not None and blur_elem.text and blur_elem.text.isdigit():
            blur_level = int(blur_elem.text)

        best_in_burst = False
        best_elem = root.find(".//tprs:best_in_burst", namespaces)
        if best_elem is not None and best_elem.text == "True":
            best_in_burst = True

        burst_id = None
        burst_id_elem = root.find(".//tprs:burst_id", namespaces)
        if burst_id_elem is not None:
            burst_id = burst_id_elem.text

        provider = None
        provider_elem = root.find(".//tprs:provider", namespaces)
        if provider_elem is not None:
            provider = provider_elem.text

        return PhotoAnalysis(
            photo_path=photo_path,
            rating=rating,
            rating_reason=rating_reason,
            keywords=keywords,
            description=description,
            primary_subject=primary_subject,
            primary_subject_bounding_box=primary_subject_bounding_box,
            raw_response=raw_response,
            best_in_burst=best_in_burst,
            blur_level=blur_level,
            burst_id=burst_id,
            provider=provider
        )
    except Exception as e:
        logger.warning(f"Failed to parse XMP {xmp_path}: {e}")
        return PhotoAnalysis(photo_path, 0, "N/A", [], f"Error loading XMP: {e}")


def select_best_in_burst(


    burst_analyses: list[PhotoAnalysis],


    client: VLMClient


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


    


    try:


        with contextlib.ExitStack() as stack:


            for c in candidates:


                # Resize for comparison to save memory and tokens


                resized = stack.enter_context(resize_image(c.photo_path, max_dimension=512))


                if resized:


                    image_paths.append(resized)


                else:


                    image_paths.append(c.photo_path)


            


            prompt = load_prompt("best_in_burst.txt")





            response = client.generate(


                prompt=prompt,


                image_paths=image_paths,


                max_tokens=100


            )


            if not response or not response.text:


                return candidates[0]





            response_text = response.text


                


            # Clean JSON


            clean_text = clean_json_response(response_text)


            


            try:


                data = json.loads(clean_text)


                best_index = int(data.get("best_index", 0))


                if 0 <= best_index < len(candidates):


                    best = candidates[best_index]


                    best.rating_reason += f"\nBest reason: {data.get('reason', 'N/A')}"


                    best.best_in_burst = True


                    return best


            except Exception as e:


                logger.warning(f"Failed to parse burst selection response: {e}")


            


    except Exception as e:


        logger.error(f"Burst selection failed: {e}")


                    


    # Fallback to first candidate (highest rated)


    return candidates[0]


def process_photos_batch(
    photos: list[Path],
    model_name: str = DEFAULT_VLM_MODEL,
    output_dir: Optional[Path] = None,
    status_callback: Optional[Callable[[int, int, Optional[Path], Optional[PhotoAnalysis]], None]] = None,
    stop_event: Optional[threading.Event] = None,
    api_base: Optional[str] = None,
    api_key: str = "lm-studio",
    num_workers: int = 1,
    provider_preferences: Optional[str] = None
) -> list[tuple[PhotoAnalysis, Path]]:
    """Process a batch of photos and generate XMP sidecars.

    Args:
        photos: List of photo paths to process.
        model_name: mlx-vlm model name.
        output_dir: Optional output directory for XMP files.
        status_callback: Callback for progress updates.
        stop_event: Optional threading.Event to signal cancellation.
        api_base: Optional API base URL. If set, local model is skipped.
        api_key: API key for custom endpoint.
        num_workers: Number of concurrent workers.

    Returns:
        List of (PhotoAnalysis, xmp_path) tuples.
    """
    results = []
    photos_to_process = []
    used_burst_ids = set()

    # Check for existing sidecars
    for photo_path in photos:
        if output_dir:
            xmp_path = output_dir / f"{photo_path.stem}.xmp"
        else:
            xmp_path = photo_path.with_suffix(".xmp")
            
        if xmp_path.exists():
            analysis = load_analysis_from_xmp(xmp_path, photo_path)
            logger.info(f"Sidecar exists for {photo_path.name}: rating {analysis.rating}, keywords {analysis.keywords}")
            results.append((analysis, xmp_path))
            if analysis.burst_id:
                used_burst_ids.add(analysis.burst_id)
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

    # Initialize VLM Client
    try:
        client = VLMClient(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            provider_preferences=provider_preferences,
            app_name="tprs"
        )
    except Exception as e:
        logger.error(f"Failed to initialize VLM Client: {e}")
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
    burst_iterator = generate_bursts(
        photos_to_process, client, 
        comparison_callback=comparison_cb
    )
    
    # Shared state for concurrency
    processed_count_lock = threading.Lock()
    used_burst_ids_lock = threading.Lock()
    # Use a mutable container for processed_count to share across threads
    shared_state = {"processed_count": processed_count}

    def process_single_burst(burst_data):
        burst, burst_keyword = burst_data
        if stop_event and stop_event.is_set():
            return []

        logger.info(f"Processing burst ({len(burst)} photos)")
        
        burst_analyses = []
        for photo_path in burst:
            if stop_event and stop_event.is_set():
                break

            # Notify start
            if status_callback:
                with processed_count_lock:
                    current_count = shared_state["processed_count"]
                try:
                    status_callback(current_count, total_photos, photo_path, None, None)
                except TypeError:
                    status_callback(current_count, total_photos, photo_path, None)

            # Analyze photo
            analysis = analyze_photo(
                photo_path, client
            )
            
            with processed_count_lock:
                shared_state["processed_count"] += 1
                current_count = shared_state["processed_count"]
            
            if analysis:
                burst_analyses.append(analysis)
                # Notify end with analysis
                if status_callback:
                    try:
                        status_callback(current_count, total_photos, photo_path, analysis, None)
                    except TypeError:
                        status_callback(current_count, total_photos, photo_path, analysis)
            else:
                # Notify end (failed)
                if status_callback:
                    try:
                        status_callback(current_count, total_photos, photo_path, None, None)
                    except TypeError:
                        status_callback(current_count, total_photos, photo_path, None)

        if not burst_analyses:
            return []

        # Determine burst_id
        if not burst_keyword:
            # Try primary subject of first photo
            if burst_analyses[0].primary_subject:
                burst_keyword = burst_analyses[0].primary_subject
            else:
                burst_keyword = "burst"
        
        # Sanitize keyword
        burst_keyword = "".join(c for c in burst_keyword if c.isalnum()).lower()
        if not burst_keyword:
            burst_keyword = "burst"
            
        # Generate unique ID
        with used_burst_ids_lock:
            counter = 1
            while True:
                candidate_id = f"{burst_keyword}_{counter:04d}"
                if candidate_id not in used_burst_ids:
                    burst_id = candidate_id
                    used_burst_ids.add(burst_id)
                    break
                counter += 1
            
        # Assign to all
        for analysis in burst_analyses:
            analysis.burst_id = burst_id

        # Select best in burst if more than 1 photo
        best_photo = None
        if len(burst_analyses) > 1:
            best_photo = select_best_in_burst(
                burst_analyses, client
            )
            
            # Apply rating constraints
            for analysis in burst_analyses:
                if analysis == best_photo:
                    analysis.keywords.append("BestInBurst")
                else:
                    if analysis.rating >= 5:
                        analysis.rating = 4
                    analysis.keywords.append("BurstDuplicate")
        
        # Generate XMP for all
        burst_results = []
        for analysis in burst_analyses:
            if output_dir:
                xmp_path = output_dir / f"{analysis.photo_path.stem}.xmp"
            else:
                xmp_path = None

            xmp_file = generate_xmp_sidecar(analysis, xmp_path)
            burst_results.append((analysis, xmp_file))
            
        return burst_results

    if num_workers > 1:
        logger.info(f"Processing bursts with {num_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all bursts
            futures = [executor.submit(process_single_burst, burst_data) for burst_data in burst_iterator]
            
            for future in concurrent.futures.as_completed(futures):
                if stop_event and stop_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    burst_results = future.result()
                    results.extend(burst_results)
                except Exception as e:
                    logger.error(f"Burst processing failed: {e}")
    else:
        # Sequential processing
        for burst_data in burst_iterator:
            if stop_event and stop_event.is_set():
                logger.info("Processing cancelled by user.")
                break
            
            burst_results = process_single_burst(burst_data)
            results.extend(burst_results)

    logger.info(f"Processed {len(results)} photos")
    return results

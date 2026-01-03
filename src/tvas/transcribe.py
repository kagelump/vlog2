#!/usr/bin/env python3
"""Transcribe video files to SRT format using mlx_whisper with Silero VAD.

This module provides transcription functionality for the TVAS pipeline.
It uses Voice Activity Detection (VAD) to identify speech segments
before transcription, reducing hallucinations from silence/noise.

The output is in SRT format for easy integration with video editing tools.
"""

import logging
import json
import math
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import os
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib"

from tvas.vad_utils import get_speech_segments, load_vad_model

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of transcribing a video file."""
    
    video_path: Path
    srt_path: Path | None
    json_path: Path | None
    success: bool
    error: str | None = None
    num_segments: int = 0
    language: str | None = None


def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def merge_transcription_segments(
    vad_segments: List[Dict],
    transcription_results: List[Dict]
) -> Dict:
    """
    Merge multiple transcription results from VAD segments into a single result.
    
    Args:
        vad_segments: List of VAD segments with 'start' and 'end' times
        transcription_results: List of transcription results (one per VAD segment)
        
    Returns:
        Merged transcription result dict
    """
    merged = {
        "text": "",
        "segments": [],
    }
    languages = set()
    
    # Check if any transcription contains Chinese - if so, initialize OpenCC for s2tw conversion
    has_chinese = any(trans_result.get('language') == 'zh' for trans_result in transcription_results)
    cc = None
    if has_chinese:
        try:
            from opencc import OpenCC
            cc = OpenCC('s2tw')
        except ImportError:
            logger.warning("OpenCC not available for Chinese conversion")
    
    for vad_seg, trans_result in zip(vad_segments, transcription_results):
        offset = vad_seg['start']
        result_ok = False
        
        # Adjust segment timestamps and add to merged result
        for segment in trans_result.get("segments", []):
            # Skip segments with NaN avg_logprob
            avg_logprob = segment.get('avg_logprob')
            if avg_logprob is not None:
                if math.isnan(avg_logprob):
                    continue
                if avg_logprob < -1.0:  # Arbitrary threshold to filter low-confidence segments
                    continue
            result_ok = True
            if segment.get('compression_ratio', 100) > 3.0:
                logger.debug('Discarding high compression ratio segment: %s', segment)
                continue
            
            adjusted_segment = segment.copy()
            adjusted_segment['start'] += offset
            adjusted_segment['end'] += offset
            
            # Convert Chinese text if OpenCC is initialized
            if cc and 'text' in adjusted_segment:
                adjusted_segment['text'] = cc.convert(adjusted_segment['text'])
            
            # Adjust word timestamps if present
            if 'words' in adjusted_segment:
                adjusted_words = []
                for word in adjusted_segment['words']:
                    adjusted_word = word.copy()
                    adjusted_word['start'] += offset
                    adjusted_word['end'] += offset
                    # Convert Chinese word text if OpenCC is initialized
                    if cc and 'word' in adjusted_word:
                        adjusted_word['word'] = cc.convert(adjusted_word['word'])
                    adjusted_words.append(adjusted_word)
                adjusted_segment['words'] = adjusted_words
            
            merged["segments"].append(adjusted_segment)
        
        if result_ok:
            # Add text with space separator
            if merged["text"]:
                merged["text"] += " "
            text_to_add = trans_result.get("text", "")
            # Convert Chinese text if OpenCC is initialized
            if cc:
                text_to_add = cc.convert(text_to_add)
            merged["text"] += text_to_add
            languages.add(trans_result.get("language", 'en'))
        
    merged["language"] = list(languages)
    
    return merged


def transcription_to_srt(transcription: Dict, output_path: Path) -> None:
    """Convert transcription result to SRT format.
    
    Args:
        transcription: Transcription result dict with 'segments'.
        output_path: Path to write SRT file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(transcription.get('segments', []), start=1):
            start_time = format_timestamp_srt(segment['start'])
            end_time = format_timestamp_srt(segment['end'])
            text = segment['text'].strip()
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n")
            f.write("\n")


def transcribe_video(
    video_path: Path,
    model: str = "mlx-community/whisper-large-v3-turbo",
    output_dir: Path | None = None,
) -> TranscriptionResult:
    """Transcribe a single video file to SRT using mlx_whisper with VAD.
    
    Args:
        video_path: Path to input video file.
        model: Model ID for mlx_whisper.
        output_dir: Directory to write output files (defaults to video directory).
        
    Returns:
        TranscriptionResult with details of the transcription.
    """
    try:
        from mlx_whisper import transcribe
    except ImportError:
        logger.error("mlx_whisper not installed. Install with: pip install mlx-whisper")
        return TranscriptionResult(
            video_path=video_path,
            srt_path=None,
            json_path=None,
            success=False,
            error="mlx_whisper not installed"
        )
    
    if output_dir is None:
        output_dir = video_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file paths
    stem = video_path.stem
    json_path = output_dir / f"{stem}_whisper.json"
    srt_path = output_dir / f"{stem}.srt"
    
    # Check for cached transcription
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                final_result = json.load(f)
            logger.info(f"Using cached transcription for {video_path.name}")
            
            # Generate SRT if it doesn't exist
            if not srt_path.exists():
                transcription_to_srt(final_result, srt_path)
                
            return TranscriptionResult(
                video_path=video_path,
                srt_path=srt_path,
                json_path=json_path,
                success=True,
                num_segments=len(final_result.get('segments', [])),
                language=final_result.get('language')
            )
        except Exception as e:
            logger.warning(f"Failed to load cached transcription from {json_path}: {e}")
    
    logger.info(f"Transcribing {video_path.name} with model {model}")
    
    try:
        # Load VAD model
        logger.info("Loading Silero VAD model...")
        vad_model, vad_utils = load_vad_model()
        
        # Detect speech segments
        logger.info("Detecting speech segments...")
        speech_segments = get_speech_segments(
            str(video_path),
            vad_model=vad_model,
            vad_utils=vad_utils
        )
        logger.info(f"Detected {len(speech_segments)} speech segments")
        
        # Transcribe based on VAD results
        if speech_segments:
            # Transcribe each speech segment separately
            logger.info(f"Transcribing {len(speech_segments)} speech segments...")
            transcription_results = []
            
            for i, segment in enumerate(speech_segments, 1):
                logger.info(f"Transcribing segment {i}/{len(speech_segments)}: "
                           f"{segment['start']:.2f}s - {segment['end']:.2f}s")
                
                # Transcribe segment with word timestamps enabled
                result = transcribe(
                    audio=str(video_path),
                    path_or_hf_repo=model,
                    verbose=None,
                    word_timestamps=True,
                    clip_timestamps=[segment['start'], segment['end']],
                    temperature=(0.0, 0.2, 0.4, 0.5)
                )
                transcription_results.append(result)
            
            # Merge results from all segments
            final_result = merge_transcription_segments(speech_segments, transcription_results)
        else:
            # No speech segments detected
            logger.info("No speech segments detected — writing empty transcription")
            final_result = {
                "text": "",
                "segments": [],
                "language": "unknown",
            }
        
        # Write JSON output
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        # Write SRT output
        transcription_to_srt(final_result, srt_path)
        
        logger.info(f"Transcription completed: {srt_path}")
        logger.info(f"  Total segments: {len(final_result.get('segments', []))}")
        logger.info(f"  Detected language: {final_result.get('language', 'unknown')}")
        
        return TranscriptionResult(
            video_path=video_path,
            srt_path=srt_path,
            json_path=json_path,
            success=True,
            num_segments=len(final_result.get('segments', [])),
            language=final_result.get('language')
        )
        
    except Exception as e:
        logger.error(f"Transcription failed for {video_path.name}: {e}")
        return TranscriptionResult(
            video_path=video_path,
            srt_path=None,
            json_path=None,
            success=False,
            error=str(e)
        )


def transcribe_clips_batch(
    clips: list[tuple[Path, Path | None]],
    model: str = "mlx-community/whisper-large-v3-turbo",
) -> list[TranscriptionResult]:
    """Transcribe a batch of video clips.
    
    Args:
        clips: List of (source_path, proxy_path) tuples. Transcribes proxy if available.
        model: Model ID for mlx_whisper.
        
    Returns:
        List of TranscriptionResult objects.
    """
    results = []
    
    for i, (source_path, proxy_path) in enumerate(clips, 1):
        # Use proxy if available, otherwise use source
        video_to_transcribe = proxy_path if proxy_path and proxy_path.exists() else source_path
        
        logger.info(f"Transcribing clip {i}/{len(clips)}: {video_to_transcribe.name}")
        
        result = transcribe_video(
            video_path=video_to_transcribe,
            model=model,
        )
        results.append(result)
    
    # Summary logging
    successful = sum(1 for r in results if r.success)
    with_speech = sum(1 for r in results if r.success and r.num_segments > 0)
    logger.info(f"Transcription complete: {successful}/{len(results)} successful, "
                f"{with_speech} with speech")
    
    return results

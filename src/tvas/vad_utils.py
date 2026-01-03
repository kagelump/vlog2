#!/usr/bin/env python3
"""
Voice Activity Detection (VAD) utilities using Silero VAD.

This module provides functions to detect speech segments in audio files
before transcription, reducing hallucination by avoiding silent regions.

Inspired by WhisperX approach but without diarization.
"""
from __future__ import annotations

import logging
import torch
import soundfile as sf
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


def load_vad_model():
    """Load Silero VAD model from torch.hub.
    
    Returns:
        tuple: (model, utils) where utils contains helper functions
    """
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        return model, utils
    except Exception as e:
        logger.error(f"Failed to load Silero VAD model: {e}")
        raise


def load_audio_with_ffmpeg(
    audio_path: str,
    sample_rate: int = 16000
) -> tuple[np.ndarray, int]:
    """
    Load audio file using ffmpeg and soundfile.
    
    This function uses ffmpeg to decode the audio to WAV format,
    avoiding the need for torchaudio's ffmpeg backend integration.
    
    Args:
        audio_path: Path to the audio/video file
        sample_rate: Target sample rate (default: 16000 Hz)
    
    Returns:
        tuple: (waveform as numpy array, sample_rate)
        waveform shape: (samples,) for mono or (samples, channels) for stereo
    """
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        temp_wav = tmp_file.name
    
    try:
        # Use ffmpeg to convert audio to WAV format at target sample rate
        # -vn: no video, -acodec pcm_s16le: 16-bit PCM, -ac 1: mono, -ar: sample rate
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', audio_path,  # Input file
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono (convert to single channel)
            temp_wav
        ]
        
        # Run ffmpeg silently
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        # Load the WAV file using soundfile
        waveform, sr = sf.read(temp_wav, dtype='float32')
        
        # Ensure waveform is 1D for mono
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        
        return waveform, sr
        
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed to process {audio_path}: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to load audio with ffmpeg: {e}")
    except Exception as e:
        logger.error(f"Failed to load audio from {audio_path}: {e}")
        raise
    finally:
        # Clean up temp file
        try:
            Path(temp_wav).unlink(missing_ok=True)
        except Exception:
            pass


def resample_audio(
    waveform: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to target sample rate using linear interpolation.
    
    Args:
        waveform: Audio waveform as numpy array
        orig_sr: Original sample rate
        target_sr: Target sample rate
    
    Returns:
        Resampled waveform
    """
    if orig_sr == target_sr:
        return waveform
    
    # Calculate new length
    duration = len(waveform) / orig_sr
    new_length = int(duration * target_sr)
    
    # Use numpy's linear interpolation
    old_indices = np.arange(len(waveform))
    new_indices = np.linspace(0, len(waveform) - 1, new_length)
    resampled = np.interp(new_indices, old_indices, waveform)
    
    return resampled.astype(np.float32)


def get_speech_segments(
    audio_path: str,
    vad_model=None,
    vad_utils=None,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    padding_duration_ms: int = 30,
) -> List[Dict[str, float]]:
    """
    Detect speech segments in an audio file using Silero VAD.
    
    Args:
        audio_path: Path to the audio file
        vad_model: Pre-loaded VAD model (optional, will load if None)
        vad_utils: VAD utilities (optional, will load if None)
        sample_rate: Target sample rate for VAD (16kHz recommended)
        threshold: Speech probability threshold (0.0-1.0)
        min_speech_duration_ms: Minimum speech segment duration in ms
        min_silence_duration_ms: Minimum silence duration between segments
        padding_duration_ms: Padding to add around speech segments
    
    Returns:
        List of dicts with 'start' and 'end' timestamps in seconds
        
    Example:
        >>> segments = get_speech_segments("audio.mp4")
        >>> print(segments)
        [{'start': 0.5, 'end': 5.2}, {'start': 6.0, 'end': 10.5}]
    """
    # Load VAD model if not provided
    if vad_model is None or vad_utils is None:
        vad_model, vad_utils = load_vad_model()
    
    get_speech_timestamps = vad_utils[0]
    
    # Load audio file
    try:
        # Load audio using ffmpeg + soundfile (avoiding torchaudio's ffmpeg backend issues)
        # waveform shape: (samples,) for mono
        waveform, loaded_sr = load_audio_with_ffmpeg(audio_path, sample_rate=sample_rate)
        
        # Convert numpy array to torch tensor for VAD model
        waveform_tensor = torch.from_numpy(waveform)
        
        # Get speech timestamps (returns sample indices)
        speech_timestamps = get_speech_timestamps(
            waveform_tensor,
            vad_model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=padding_duration_ms,
        )
        
        # Convert from sample indices to seconds
        segments = []
        for timestamp in speech_timestamps:
            start_sec = timestamp['start'] / sample_rate
            end_sec = timestamp['end'] / sample_rate
            segments.append({
                'start': start_sec,
                'end': end_sec
            })
        
        logger.info(f"Detected {len(segments)} speech segments in {audio_path}")
        return segments
        
    except Exception as e:
        logger.error(f"Failed to detect speech segments in {audio_path}: {e}")
        raise e


def extract_audio_segment(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str | None = None,
    sample_rate: int = 16000
) -> str:
    """
    Extract a segment from an audio file.
    
    Args:
        audio_path: Path to source audio file
        start_sec: Start time in seconds
        end_sec: End time in seconds
        output_path: Path to save extracted segment (optional)
        sample_rate: Target sample rate
        
    Returns:
        Path to the extracted audio segment
    """
    try:
        # Load audio using ffmpeg + soundfile
        waveform, loaded_sr = load_audio_with_ffmpeg(audio_path, sample_rate=sample_rate)
        
        # Calculate sample indices
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        
        # Extract segment
        segment = waveform[start_sample:end_sample]
        
        # Save to file if output_path provided
        if output_path:
            sf.write(output_path, segment, sample_rate)
            return output_path
        else:
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.wav',
                prefix='vad_segment_'
            )
            sf.write(temp_file.name, segment, sample_rate)
            return temp_file.name
            
    except Exception as e:
        logger.error(f"Failed to extract audio segment: {e}")
        raise


if __name__ == "__main__":
    # Test VAD functionality
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vad_utils.py <audio_file>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    audio_file = sys.argv[1]
    segments = get_speech_segments(audio_file)
    
    print(f"\nDetected {len(segments)} speech segments:")
    for i, seg in enumerate(segments, 1):
        duration = seg['end'] - seg['start']
        print(f"  {i}. {seg['start']:.2f}s - {seg['end']:.2f}s (duration: {duration:.2f}s)")

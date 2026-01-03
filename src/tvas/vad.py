from __future__ import annotations

import sys
from pathlib import Path
from silero_vad import load_silero_vad, get_speech_timestamps
import json
import logging
import torch
import soundfile as sf
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict


logger = logging.getLogger(__name__)


class SpeechDetector:
    """Detect speech timestamps in audio/video files using Silero VAD."""
    
    DEFAULT_SAMPLE_RATE = 16000
    
    def __init__(self):
        """Initialize the speech detector with the Silero VAD model."""
        self.model = load_silero_vad()
    
    def load_audio_with_ffmpeg(
        self,
        audio_path: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> tuple[np.ndarray, int]:
        """
        Load audio file using ffmpeg and soundfile.
        
        This method uses ffmpeg to decode the audio to WAV format,
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
    
    def detect_speech(self, audio_path: str) -> List[Dict[str, float]]:
        """
        Detect speech timestamps in an audio/video file.
        
        Args:
            audio_path: Path to the audio/video file
        
        Returns:
            List of speech timestamp dictionaries with 'start' and 'end' keys (in seconds)
        """
        # Load audio using ffmpeg + soundfile
        waveform, loaded_sr = self.load_audio_with_ffmpeg(audio_path, sample_rate=self.DEFAULT_SAMPLE_RATE)
        
        # Convert numpy array to torch tensor for VAD model
        waveform_tensor = torch.from_numpy(waveform)
        
        # Get speech timestamps in seconds
        speech_timestamps = get_speech_timestamps(
            waveform_tensor,
            self.model,
            return_seconds=True,
        )
        
        return speech_timestamps


def main():
    """CLI entry point for speech detection."""
    if len(sys.argv) < 2:
        print("Usage: python vad.py <audio_file>", file=sys.stderr)
        sys.exit(2)

    input_path = sys.argv[1]
    # ensure path exists
    p = Path(input_path)
    if not p.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    detector = SpeechDetector()
    speech_timestamps = detector.detect_speech(input_path)
    
    print(json.dumps(speech_timestamps))


if __name__ == "__main__":
    main()
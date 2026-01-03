#!/usr/bin/env python3
"""Transcribe a single preview video to text using mlx_whisper.

The output is in text format (_whisper.txt) with [timestamp] line format.

This script is intended to be run via Snakemake's `script:` directive where
the `snakemake` object is available, or run standalone from the CLI.

When run by Snakemake, it will read parameters from `snakemake.params` and
`snakemake.input`/`snakemake.wildcards`.

CLI usage:
    python transcribe.py --model <model> --input <path>

Author: automated migration
"""

import logging
import sys
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

# Default logging configuration for CLI runs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from mlx_whisper import transcribe
from opencc import OpenCC

LIKELY_SILENCE_TEXT = [
    "thank you",
]
LIKELY_SILENCE_LOGPROB_THRESHOLD = -0.5

def segment_is_ok(segment: Dict) -> bool:
    """Check if a transcription segment is valid (non-empty text)."""
    text = segment.get('text', '').strip()
    if not len(text):
        return False
    if segment.get('compression_ratio', 0.0) > 2.0:
        return False
    if segment.get('avg_logprob', 0.0) < -1.0:
        return False
    for silence_text in LIKELY_SILENCE_TEXT:
        if silence_text in text.lower() and segment.get('avg_logprob', 0.0) < LIKELY_SILENCE_LOGPROB_THRESHOLD:
            return False
    print(f"DEBUG: segment = '{segment}'")
    return True


def run_transcribe(
    model: str,
    input_path: str,
) -> Optional[str]:
    """Transcribe a single preview file using mlx_whisper Python API.

    Args:
        model: Model ID for mlx_whisper
        input_path: Path to input video/audio file

    Returns:
        Transcription text on success, None if no speech detected or failure.
    """

    logging.info("Running VAD to check for speech segments: %s", input_path)
    
    try:
        # Run VAD as subprocess to check for speech segments
        vad_script = Path(__file__).parent / "vad.py"
        result = subprocess.run(
            [sys.executable, str(vad_script), input_path],
            capture_output=True,
            text=True,
            check=True,
        )
        
        speech_timestamps = json.loads(result.stdout.strip())
        
        if not speech_timestamps:
            logging.info("No speech segments detected. Skipping transcription.")
            return None
        
        logging.info("Found %d speech segments", len(speech_timestamps))
        
    except subprocess.CalledProcessError as e:
        logging.error("VAD failed: %s", e.stderr)
        return None
    except json.JSONDecodeError as e:
        logging.error("Failed to parse VAD output: %s", e)
        return None
    except Exception as e:
        logging.error("Error running VAD: %s", e)
        return None
    
    logging.info("Transcribing %s with model %s", input_path, model)
    
    try:
        # Transcribe directly
        result = transcribe(
            audio=input_path,
            path_or_hf_repo=model,
            verbose=None,
            word_timestamps=True,  # Enable for richer metadata
            temperature=(0.0, 0.2, 0.4, 0.5)
        )
        
        # Post-process for Chinese conversion
        if result.get('language') == 'zh':
            cc = OpenCC('s2tw')
            if 'text' in result:
                result['text'] = cc.convert(result['text'])
            
            for segment in result.get('segments', []):
                if 'text' in segment:
                    segment['text'] = cc.convert(segment['text'])
                
                if 'words' in segment:
                    for word in segment['words']:
                        if 'word' in word:
                            word['word'] = cc.convert(word['word'])

        formatted_lines = []
        for segment in result.get('segments', []):
            if not segment_is_ok(segment):
                continue
            start = segment.get('start', 0.0)
            text = segment.get('text', '').strip()
            
            # Format timestamp as MM:SS or HH:MM:SS
            h = int(start // 3600)
            m = int((start % 3600) // 60)
            s = int(start % 60)
            
            if h > 0:
                timestamp = f"{h:02d}:{m:02d}:{s:02d}"
            else:
                timestamp = f"{m:02d}:{s:02d}"
            
            formatted_lines.append(f"[{timestamp}] {text}")
        
        logging.info(f"  Total segments: {len(result.get('segments', []))}")
        logging.info(f"  Detected language: {result.get('language', 'unknown')}")
        
        return "\n".join(formatted_lines)
        
    except Exception as e:
        logging.exception("Error during transcription: %s", e)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe a single preview video to text using mlx_whisper"
    )
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-turbo", help="Model id for mlx_whisper")
    parser.add_argument("--input", required=True, help="Input video path")

    args = parser.parse_args()
    transcription = run_transcribe(args.model, args.input)
    
    if transcription:
        # Determine output path (same directory as input, stem + _whisper.txt)
        p = Path(args.input)
        stem = p.stem
        output_file = p.parent / f"{stem}_whisper.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcription + "\n")
        
        logging.info("Transcription completed: %s", output_file)
        print(transcription)
        sys.exit(0)
    else:
        sys.exit(1)
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
from pathlib import Path
from typing import Dict

import os
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib"

from mlx_whisper import transcribe
from opencc import OpenCC

def segment_is_ok(segment: Dict) -> bool:
    """Check if a transcription segment is valid (non-empty text)."""
    text = segment.get('text', '').strip()
    if not len(text):
        return False
    if segment.get('compression_ratio', 0.0) > 2.0:
        return False
    return True


def run_transcribe(
    model: str,
    input_path: str,
) -> int:
    """Transcribe a single preview file using mlx_whisper Python API.

    The output text file will be written to the same directory as the input file
    and will use the input file stem with suffix `_whisper.txt`.

    Args:
        model: Model ID for mlx_whisper
        input_path: Path to input video/audio file

    Returns:
        0 on success, non-zero on failure.
    """
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

        # Determine output path (same directory as input, stem + _whisper.txt)
        p = Path(input_path)
        stem = p.stem
        output_file = p.parent / f"{stem}_whisper.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
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
                
                f.write(f"[{timestamp}] {text}\n")
        
        logging.info("Transcription completed: %s", output_file)
        logging.info(f"  Total segments: {len(result.get('segments', []))}")
        logging.info(f"  Detected language: {result.get('language', 'unknown')}")
        return 0
        
    except Exception as e:
        logging.exception("Error during transcription: %s", e)
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe a single preview video to text using mlx_whisper"
    )
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-turbo", help="Model id for mlx_whisper")
    parser.add_argument("--input", required=True, help="Input video path")

    args = parser.parse_args()
    sys.exit(run_transcribe(args.model, args.input))
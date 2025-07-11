import os
import subprocess
from config import DATA_DIR

"""
Batch-convert all audio in data/raw with extensions
['.amr', '.mp3', '.mpeg', '.flac'] → 16 kHz mono .wav
Requires ffmpeg on your PATH.
"""

RAW_DIR = os.path.join(DATA_DIR, 'raw')
TO_CONVERT = ['.amr', '.mp3', '.mpeg', '.flac']

for root, _, files in os.walk(RAW_DIR):
    for fname in files:
        base, ext = os.path.splitext(fname)
        if ext.lower() in TO_CONVERT:
            src = os.path.join(root, fname)
            dst = os.path.join(root, base + '.wav')
            if not os.path.exists(dst):
                print(f"Converting: {src} → {dst}")
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', src,
                    '-ar', '16000',  # 16 kHz
                    '-ac', '1',      # mono
                    dst
                ], check=True)
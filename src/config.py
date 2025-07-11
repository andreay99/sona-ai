import os
import torch

# ── BASE PATHS ────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")

# where raw audio lives
RAW_DIR       = os.path.join(DATA_DIR, "raw")

# where your train/val/test split CSVs go
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR    = PROCESSED_DIR

# where extracted mel‐spectrograms & their CSV indexes go
FEATURES_DIR  = os.path.join(PROCESSED_DIR, "features")

# ── AUDIO / FEATURES ──────────────────────────────────────────────────────────
SR           = 16_000
MAX_DURATION = 4.0
N_MELS       = 64
HOP_LENGTH   = 512

# ── TRAINING ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS        = 50
LEARNING_RATE = 1e-3
import os

# project root
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RAW_DIR     = os.path.join(DATA_DIR, "raw")
SPLITS_DIR  = os.path.join(DATA_DIR, "splits")
FEATURES_DIR= os.path.join(DATA_DIR, "features")
MODELS_DIR  = os.path.join(DATA_DIR, "models")

# audio settings
SR           = 16_000      # sample rate
MAX_DURATION = 4.0         # seconds
N_MELS       = 64
HOP_LENGTH   = 512
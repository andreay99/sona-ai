import os
import numpy as np
import pandas as pd
import librosa

from .config import SPLITS_DIR, FEATURES_DIR, SR, MAX_DURATION, N_MELS, HOP_LENGTH

def extract_split(split: str):
    csv_path = os.path.join(SPLITS_DIR, f"{split}.csv")
    df       = pd.read_csv(csv_path)

    # find columns
    fp_col = next((c for c in df.columns if c.lower() in ("filepath","file_path","path","filename")), None)
    lb_col = next((c for c in df.columns if c.lower() in ("label","emotion")), None)
    if fp_col is None or lb_col is None:
        raise ValueError(f"Expected filepath+label in {csv_path}, got {df.columns.tolist()}")

    out_dir = os.path.join(FEATURES_DIR, split)
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for _, row in df.iterrows():
        src_fp = row[fp_col]
        label  = row[lb_col]
        stem   = os.path.splitext(os.path.basename(src_fp))[0]
        out_fp = os.path.join(out_dir, stem + ".npy")

        if not os.path.exists(src_fp):
            print(f"⚠️  missing file, skipping {src_fp}")
            continue

        y, _ = librosa.load(src_fp, sr=SR, duration=MAX_DURATION)
        mels = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
        np.save(out_fp, mels.astype(np.float32))
        records.append({"feature_path": out_fp, "label": label})

    feats_df = pd.DataFrame(records)
    feats_csv = os.path.join(FEATURES_DIR, f"{split}_features.csv")
    feats_df.to_csv(feats_csv, index=False)
    print(f"✔️  Extracted {len(records)} features for '{split}'")

def main():
    for split in ("train","val","test"):
        extract_split(split)

if __name__ == "__main__":
    main()
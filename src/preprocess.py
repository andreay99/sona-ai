import os, glob
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RAW_DIR, SPLITS_DIR

# make sure the splits directory exists
os.makedirs(SPLITS_DIR, exist_ok=True)

def main():
    exts = ("wav", "mp3", "mpeg", "amr")
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(RAW_DIR, "**", f"*.{ext}"), recursive=True)

    # build a DataFrame of path→label
    df = pd.DataFrame({
        "filepath": files,
        "label":    [os.path.basename(fp).split(".")[0] for fp in files]
    })

    # stratified train/val/test split
    train, rest = train_test_split(df, stratify=df.label, test_size=0.3, random_state=42)
    val,   test = train_test_split(rest, stratify=rest.label, test_size=0.5, random_state=42)

    # write to data/processed/splits/
    train.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    val.to_csv(  os.path.join(SPLITS_DIR, "val.csv"),   index=False)
    test.to_csv( os.path.join(SPLITS_DIR, "test.csv"),  index=False)

    print(f"Wrote {len(train)} train, {len(val)} val, {len(test)} test examples")

if __name__ == "__main__":
    main()
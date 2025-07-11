import os
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, SPLITS_DIR

def main():
    # assume your original metadata is in data/raw/metadata.csv
    raw_meta = os.path.join(DATA_DIR, 'raw', 'metadata.csv')
    df = pd.read_csv(raw_meta)

    # filter based on actual_files.txt and listed_in_meta.txt
    files_txt  = os.path.join(DATA_DIR, 'raw', 'actual_files.txt')
    listed_txt = os.path.join(DATA_DIR, 'raw', 'listed_in_meta.txt')
    if os.path.exists(files_txt) and os.path.exists(listed_txt):
        with open(files_txt) as f:
            actual_files = {os.path.basename(line.strip()) for line in f if line.strip()}
        with open(listed_txt) as f:
            listed_files = {os.path.basename(line.strip()) for line in f if line.strip()}
        # keep only rows whose filename appears in both lists
        df = df[df['filename'].isin(actual_files & listed_files)]

    # rename to the exact names features.py/train.py expect
    df = df.rename(columns={'file_path': 'filename', 'label': 'emotion'})

    # ensure splits folder exists
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # stratify only if possible
    try:
        train_val, test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['emotion']
        )
        train, val = train_test_split(
            train_val, test_size=0.2, random_state=42, stratify=train_val['emotion']
        )
    except ValueError:
        # fallback to plain split if stratify fails
        train_val, test = train_test_split(df, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.2, random_state=42)

    train.to_csv(os.path.join(SPLITS_DIR, 'train.csv'), index=False)
    val.to_csv(  os.path.join(SPLITS_DIR, 'val.csv'),   index=False)
    test.to_csv( os.path.join(SPLITS_DIR, 'test.csv'),  index=False)

if __name__ == '__main__':
    main()
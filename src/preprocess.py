import os
import glob
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

BASE        = os.path.dirname(__file__)
DATA_RAW    = os.path.join(BASE, '..', 'data', 'raw')
DATA_PROC   = os.path.join(BASE, '..', 'data', 'processed')
FEATURE_DIR = os.path.join(DATA_PROC, 'features')

# make sure processed dirs exist
os.makedirs(FEATURE_DIR, exist_ok=True)

def extract_features(wav_path, sr=16000, n_mels=128):
    y, _ = librosa.load(wav_path, sr=sr)
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logm = librosa.power_to_db(mels)
    return logm.T  # shape (T, n_mels)

def main():
    # 1) gather all WAVs
    wav_files = glob.glob(os.path.join(DATA_RAW, '**', '*.wav'), recursive=True)

    records = []
    for wav in wav_files:
        # 2) extract + save feature
        feat = extract_features(wav)
        name = os.path.splitext(os.path.basename(wav))[0]
        feat_path = os.path.join(FEATURE_DIR, f'{name}.npy')
        np.save(feat_path, feat)

        # 3) derive label from filename (e.g. "happy_01.wav" → "happy")
        label = name.split('_')[0]
        records.append({'feature_path': feat_path, 'label': label})

    # 4) build DataFrame & split
    df = pd.DataFrame(records)
    train, rest = train_test_split(df, stratify=df.label, test_size=0.3, random_state=42)
    val, test   = train_test_split(rest, stratify=rest.label, test_size=0.5, random_state=42)

    # 5) write CSVs
    train.to_csv(os.path.join(DATA_PROC, 'train.csv'), index=False)
    val.  to_csv(os.path.join(DATA_PROC, 'val.csv'),   index=False)
    test. to_csv(os.path.join(DATA_PROC, 'test.csv'),  index=False)

    print(f'Wrote {len(train)} train, {len(val)} val, {len(test)} test examples')

if __name__ == '__main__':
    main()
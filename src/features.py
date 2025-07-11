import os
import glob
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

# == configuration ==
BASE         = os.path.dirname(__file__)
DATA_RAW     = os.path.join(BASE, '..', 'data', 'raw')
DATA_PROC    = os.path.join(BASE, '..', 'data', 'processed')
FEATURE_DIR  = os.path.join(DATA_PROC, 'features')
SR           = 16000        # sample rate
N_MELS       = 128          # number of Mel bins
HOP_LENGTH   = 512          # hop length for spectrogram
MAX_DURATION = 3.0          # seconds
# =================

os.makedirs(FEATURE_DIR, exist_ok=True)

def extract_features(wav_path: str):
    # 1) load up to MAX_DURATION seconds
    y, _ = librosa.load(wav_path, sr=SR, duration=MAX_DURATION)

    # 2) build Mel-spectrogram
    mels = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )
    logm = librosa.power_to_db(mels)  # shape (n_mels, T)

    # 3) pad or truncate time axis to fixed T_max
    T_max = int(np.ceil(MAX_DURATION * SR / HOP_LENGTH))
    if logm.shape[1] < T_max:
        pad_amount = T_max - logm.shape[1]
        logm = np.pad(logm,
                      pad_width=((0, 0), (0, pad_amount)),
                      mode='constant',
                      constant_values=0.0)
    else:
        logm = logm[:, :T_max]

    # 4) return as (T_max, n_mels)
    return logm.T

def main():
    # gather all WAVs
    wav_files = glob.glob(os.path.join(DATA_RAW, '**', '*.wav'), recursive=True)

    records = []
    for wav in wav_files:
        feat = extract_features(wav)              # shape (T_max, n_mels)
        name = os.path.splitext(os.path.basename(wav))[0]
        feat_path = os.path.join(FEATURE_DIR, f'{name}.npy')
        np.save(feat_path, feat.astype(np.float32))

        label = name.split('_')[0]
        records.append({'feature_path': feat_path, 'label': label})

    # split into train/val/test
    df = pd.DataFrame(records)
    train, rest = train_test_split(df, stratify=df.label, test_size=0.3, random_state=42)
    val, test   = train_test_split(rest, stratify=rest.label, test_size=0.5, random_state=42)

    # write CSVs
    os.makedirs(DATA_PROC, exist_ok=True)
    train.to_csv(os.path.join(DATA_PROC, 'train.csv'), index=False)
    val.  to_csv(os.path.join(DATA_PROC, 'val.csv'),   index=False)
    test. to_csv(os.path.join(DATA_PROC, 'test.csv'),  index=False)

    print(f'Wrote {len(train)} train, {len(val)} val, {len(test)} test examples')

if __name__ == '__main__':
    main()
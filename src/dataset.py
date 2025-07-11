# src/dataset.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SERDataset(Dataset):
    """
    A PyTorch Dataset for speech emotion recognition features saved as .npy files.
    Expects a CSV with columns: feature_path,label
    """

    def __init__(self, csv_path: str, label_map: dict = None):
        """
        csv_path: path to train/val/test CSV (feature_path,label)
        label_map: optional dict mapping label string -> integer.  
                   If None, will be built from the CSV.
        """
        self.df = pd.read_csv(csv_path)

        # build or validate label→index mapping
        if label_map is None:
            labels = sorted(self.df['label'].unique())
            self.label2idx = {lab: idx for idx, lab in enumerate(labels)}
        else:
            self.label2idx = label_map

        # map string labels to ints
        self.df['label_idx'] = self.df['label'].map(self.label2idx)

        # sanity check
        if self.df['label_idx'].isnull().any():
            missing = set(self.df.loc[self.df['label_idx'].isnull(), 'label'])
            raise ValueError(f"Found labels not in label_map: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_path = row['feature_path']
        label_idx = int(row['label_idx'])

        # load the spectrogram (T, n_mels)
        feat = np.load(feat_path)

        # return (numpy array, int)
        return feat, label_idx

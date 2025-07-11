import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class SERDataset(Dataset):
    def __init__(self, features_csv: str):
        """
        features_csv should be the *_features.csv that your
        features.py generated (with columns `feature_path,label`).
        """
        self.df = pd.read_csv(features_csv)
        # detect the columns
        if 'feature_path' not in self.df.columns or 'label' not in self.df.columns:
            raise ValueError(f"Expected columns ['feature_path','label'] in {features_csv} but got {self.df.columns.tolist()}")

        # build a label→index map
        self.labels = sorted(self.df['label'].unique())
        self.label2idx = {lab: i for i, lab in enumerate(self.labels)}

        self.fp_col = 'feature_path'
        self.lb_col = 'label'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = np.load(row[self.fp_col])             # (n_mels, T)
        label_str = row[self.lb_col]
        label = self.label2idx[label_str]
        return feat, label
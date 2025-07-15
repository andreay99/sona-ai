import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SERDataset(Dataset):
    def __init__(self, features_csv: str):
        df = pd.read_csv(features_csv)
        self.filepaths = df['feature_path'].tolist()
        self.labels    = df['label'].tolist()
        # build label→int mapping
        classes = sorted(set(self.labels))
        self.label2idx = {lab:i for i,lab in enumerate(classes)}

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # load (n_mels, time_steps)
        mel = np.load(self.filepaths[idx])
        # to torch tensor and float
        mel_t = torch.from_numpy(mel).float()
        # add channel dim → (1, n_mels, time_steps)
        mel_t = mel_t.unsqueeze(0)
        # convert label to int
        label = self.label2idx[self.labels[idx]]
        return mel_t, label
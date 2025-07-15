# src/evaluate.py
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import DEVICE
from src.model import MyModel

def main():
    p = argparse.ArgumentParser(description="Evaluate SER model on a features CSV")
    p.add_argument("features_csv", help="Path to *_features.csv")
    p.add_argument("model_path",    help="Path to your .pth checkpoint")
    args = p.parse_args()

    # 1) load features CSV
    df = pd.read_csv(args.features_csv)
    # get sorted list of classes
    labels = sorted(df.label.unique())
    label2idx = {l:i for i,l in enumerate(labels)}

    # 2) build & load your model
    model = MyModel(num_classes=len(labels)).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []

    # 3) for each row, load the mel-spectrogram and predict
    with torch.no_grad():
        for _, row in df.iterrows():
            feat = np.load(row.feature_path)             # (n_mels, T)
            x = torch.tensor(feat, dtype=torch.float32)  # -> tensor
            x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)   # (1,1,n_mels,T)
            logits = model(x)
            pred = logits.argmax(dim=1).item()

            y_true.append(label2idx[row.label])
            y_pred.append(pred)

    # 4) print metrics
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import joblib

from config import FEATURES_DIR, SPLITS_DIR, MODELS_DIR

def load_split(split: str):
    csv_path = os.path.join(SPLITS_DIR, f"{split}.csv")
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        lbl_col = "label"
    elif "emotion" in df.columns:
        lbl_col = "emotion"
    else:
        raise RuntimeError("CSV must have a 'label' or 'emotion' column")
    X = np.vstack([
        np.load(os.path.join(FEATURES_DIR, split, os.path.splitext(row["filename"])[0] + ".npy"))
        for _, row in df.iterrows()
    ])
    y = df[lbl_col].values
    return X, y

def main():
    data = joblib.load(os.path.join(MODELS_DIR, "logreg.pkl"))
    scaler, clf = data["scaler"], data["clf"]

    for split in ["val", "test"]:
        X, y = load_split(split)
        Xs = scaler.transform(X)
        y_pred = clf.predict(Xs)
        print(f"— {split.upper()} RESULTS —")
        print(classification_report(y, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
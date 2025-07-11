import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .dataset import SERDataset
from .model   import MyModel
from .config  import FEATURES_DIR, BATCH_SIZE, DEVICE, EPOCHS, LEARNING_RATE

def collate_fn(batch):
    feats, labels = zip(*batch)
    # feats: list of np.ndarray shape (n_mels, T_i)
    tensors = [torch.from_numpy(f.astype('float32')) for f in feats]
    # pad along time dim to longest in batch → (B, n_mels, T_max)
    padded = pad_sequence(tensors, batch_first=True)  # (B, T_max, n_mels)
    # reshape to (B,1,n_mels, T_max) for 2D CNN
    padded = padded.transpose(1, 2).unsqueeze(1)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded.to(DEVICE), labels.to(DEVICE)

def train():
    # point at the feature‐list CSVs
    train_csv = os.path.join(FEATURES_DIR, "train_features.csv")
    val_csv   = os.path.join(FEATURES_DIR, "val_features.csv")

    train_ds = SERDataset(train_csv)
    val_ds   = SERDataset(val_csv)

    # now we know how many classes we have
    num_classes = len(train_ds.labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, collate_fn=collate_fn)

    model = MyModel(num_classes=num_classes).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS+1):
        # — training —
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optim.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optim.step()
            total_loss += loss.item() * X_batch.size(0)

        # — validation —
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total   += y_batch.size(0)

        print(f"Epoch {epoch}/{EPOCHS} — "
              f"train_loss: {total_loss/len(train_ds):.4f}  "
              f"val_acc: {correct/total:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ser_cnn.pth")
    print("✔️  Model saved to models/ser_cnn.pth")

if __name__ == "__main__":
    train()
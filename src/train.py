# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import FEATURES_DIR, BATCH_SIZE, DEVICE, EPOCHS, LEARNING_RATE
from .dataset import SERDataset
from .model import MyModel

def train():
    # --- 1) Paths to your feature‐CSV files
    train_csv = os.path.join(FEATURES_DIR, "train_features.csv")
    val_csv   = os.path.join(FEATURES_DIR, "val_features.csv")

    # --- 2) Build datasets + loaders
    train_ds = SERDataset(train_csv)
    val_ds   = SERDataset(val_csv)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # --- 3) Figure out how many emotions (classes) you have
    unique_labels = sorted(train_ds.label2idx.keys())
    num_classes   = len(unique_labels)

    # --- 4) Instantiate your model, loss, optimizer
    model     = MyModel(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5) Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        # --- 6) Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
        val_acc = correct / len(val_loader.dataset)

        print(f"Epoch {epoch}/{EPOCHS} — train_loss: {avg_loss:.4f}  val_acc: {val_acc:.4f}")

    # --- 7) Save the trained weights
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "ser_cnn.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✔️  Model saved to {save_path}")

if __name__ == "__main__":
    train()
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # 1 input channel (mel spectrogram), 32→64 feature maps
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        # collapse whatever spatial size remains to 1×1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # now it's always 64×1×1 → a 64-dim vector
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.global_pool(x)      # → (B, 64, 1, 1)
        x = x.view(x.size(0), -1)    # → (B, 64)
        x = torch.relu(self.fc1(x))  # → (B, 128)
        x = self.dropout(x)
        x = self.fc2(x)              # → (B, num_classes)
        return x
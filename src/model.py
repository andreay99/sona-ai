import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # 1 input channel (the mel-spectrogram), 32→64 filters
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout     = nn.Dropout(0.5)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        # after pooling & conv2 you have 64 channels × 1 × 1 spatially
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B,1,n_mels,T)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(x)          # → (B,64,1,1)
        x = x.view(x.size(0), -1)        # → (B,64)
        x = self.dropout(x)
        return self.fc(x)                # → (B,num_classes)
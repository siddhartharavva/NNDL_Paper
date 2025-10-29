
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicScratchCNN(nn.Module):
    """Simple CNN for calcium imaging denoising from scratch"""
    def __init__(self, in_channels=1, num_layers=4, num_features=32):
        super().__init__()
        layers = []
        features = num_features
        layers.append(nn.Conv2d(in_channels, features, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, in_channels, kernel_size=3, padding=1))
        self.cnn = nn.Sequential(*layers)
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B*T, C, H, W)
        output = self.cnn(x_reshaped)
        output = output.view(B, T, C, H, W)
        return output
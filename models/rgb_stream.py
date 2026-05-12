"""RGB Stream — EfficientNet-B4 spatial feature extractor."""

import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class RGBStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.feat_dim = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        features = self.backbone.extract_features(x)
        return self.pool(features).flatten(1)

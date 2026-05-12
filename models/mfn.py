"""
Multi-stream Fusion Network (MFN) for Deepfake Detection.
EfficientNet-B4 (RGB) + ResNet18 (frequency) + multi-head attention fusion.
"""

import torch
import torch.nn as nn
import timm
from efficientnet_pytorch import EfficientNet


class M2TRModel(nn.Module):
    def __init__(self, num_classes=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.rgb_model = EfficientNet.from_pretrained('efficientnet-b4')
        rgb_feat_dim = self.rgb_model._fc.in_features
        self.rgb_model._fc = nn.Identity()

        self.dct_model = timm.create_model(
            'resnet18', pretrained=True, num_classes=0, global_pool='avg'
        )
        dct_feat_dim = self.dct_model.num_features

        fusion_dim = rgb_feat_dim + dct_feat_dim
        self.att = nn.MultiheadAttention(fusion_dim, num_heads=num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_rgb, x_dct):
        x1 = self.rgb_model.extract_features(x_rgb)
        x1 = nn.AdaptiveAvgPool2d(1)(x1).flatten(1)
        x2 = self.dct_model(x_dct)
        x = torch.cat([x1, x2], dim=1).unsqueeze(1)
        x_att, _ = self.att(x, x, x)
        x = x_att.squeeze(1)
        return self.classifier(x)

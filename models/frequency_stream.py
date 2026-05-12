"""Frequency Stream — ResNet18 DCT domain feature extractor."""

import torch.nn as nn
import timm


class FrequencyStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'resnet18', pretrained=True, num_classes=0, global_pool='avg'
        )
        self.feat_dim = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)

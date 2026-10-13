import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PneumoniaDenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.features.denseblock4.parameters():
            p.requires_grad = True
        for p in self.backbone.features.norm5.parameters():
            p.requires_grad = True
        n = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 1))

    def forward(self, x):
        features = self.backbone.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return self.backbone.classifier(out.flatten(1))

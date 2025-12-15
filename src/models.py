"""Models for image classification."""

import torch.nn as nn
from torchvision import models

class TailModel(nn.Module):
    def __init__(self, num_classes: int, dropout: float):
        super(TailModel, self).__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.linear1 = nn.Linear(in_features, 7)
        # self.activation = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(64, num_classes)
                           
    def forward(self, x):
        x = self.backbone(x)
        x = self.linear1(x)
        # x = self.activation(x)
        # x = self.dropout(x)
        # x = self.linear2(x)
        return x
import torch
import torch.nn as nn
from torchvision import models

class BaseClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.0):
        super(BaseClassifier, self).__init__()
        self.dropout_rate = dropout_rate
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.target_layer = self.model.layer4[-1]
        feature_extractor_layers = list(self.model.children())[:-1]
        if self.dropout_rate > 0:
            try:
                avgpool_idx = [i for i, layer in enumerate(feature_extractor_layers) if isinstance(layer, nn.AdaptiveAvgPool2d)][0]
                feature_extractor_layers.insert(avgpool_idx + 1, nn.Dropout(p=self.dropout_rate))
                print(f"Added Dropout layer with rate {self.dropout_rate} to feature extractor.")
            except IndexError:
                print("Warning: AdaptiveAvgPool2d not found. Adding Dropout after all conv layers.")
                feature_extractor_layers.append(nn.Dropout(p=self.dropout_rate))
        self.feature_extractor = nn.Sequential(*feature_extractor_layers)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        output = self.model.fc(features)
        return output

    def get_features(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = torch.flatten(features, 1)
        return features

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, epsilon=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - epsilon
        self.epsilon = epsilon
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.epsilon / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
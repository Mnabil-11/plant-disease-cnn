"""Models for classifying plant leaf images: a small CNN and a pretrained ResNet18."""

import torch
from torch import nn
from torchvision import models


class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_resnet18(num_classes: int, freeze_backbone: bool):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name: str, num_classes: int):
    if name == "cnn":
        return PlantDiseaseCNN(num_classes)
    if name == "resnet18":
        return build_resnet18(num_classes, freeze_backbone=True)
    if name == "resnet18_ft":
        return build_resnet18(num_classes, freeze_backbone=False)
    raise ValueError(f"unknown model: {name}")


if __name__ == "__main__":
    for name, image_size in [("cnn", 128), ("resnet18", 224), ("resnet18_ft", 224)]:
        model = build_model(name, num_classes=3)

        dummy_input = torch.randn(1, 3, image_size, image_size)
        output = model(dummy_input)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"[{name}]")
        print("  input shape:         ", tuple(dummy_input.shape))
        print("  output shape:        ", tuple(output.shape))
        print(f"  total parameters:     {total:,}")
        print(f"  trainable parameters: {trainable:,}")

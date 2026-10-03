"""Predict the disease class of a single leaf image."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from dataset import DATA_DIR, build_transforms
from model import build_model
from train import DEVICE, checkpoint_path, parse_model_arg


def load_class_names():
    # ImageFolder assigns label indices in sorted folder-name order
    return sorted(p.name for p in (DATA_DIR / "train").iterdir() if p.is_dir())


def load_model(model_name, num_classes, strong_augment=False):
    model = build_model(model_name, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path(model_name, strong_augment), map_location=DEVICE))
    model.eval()
    return model


def predict(model, model_name, image: Image.Image, class_names):
    _, eval_transform = build_transforms(model_name)
    tensor = eval_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    return {name: probabilities[i].item() for i, name in enumerate(class_names)}


def main():
    parser = parse_model_arg(__doc__)
    parser.add_argument("image_path", type=Path)
    args = parser.parse_args()

    class_names = load_class_names()
    model = load_model(args.model, len(class_names))
    probabilities = predict(model, args.model, Image.open(args.image_path), class_names)

    best_class = max(probabilities, key=probabilities.get)
    print(f"model: {args.model}")
    print(f"image: {args.image_path.name}")
    print(f"prediction: {best_class} ({probabilities[best_class]:.1%} confidence)\n")
    for name, prob in sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:<14} {prob:.1%}")


if __name__ == "__main__":
    main()

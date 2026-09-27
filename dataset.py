"""Dataset and DataLoader setup for the plant disease classifier."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent / "data"
BATCH_SIZE = 32

# Pretrained models must see inputs preprocessed the same way as during pretraining
IMAGENET_CONFIG = {
    "image_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

MODEL_CONFIGS = {
    "cnn": {
        "image_size": 128,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
    "resnet18": IMAGENET_CONFIG,
    "resnet18_ft": IMAGENET_CONFIG,
}


def build_transforms(model_name: str):
    config = MODEL_CONFIGS[model_name]
    size = (config["image_size"], config["image_size"])
    normalize = transforms.Normalize(config["mean"], config["std"])

    train_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def get_dataloaders(model_name: str = "cnn"):
    train_transform, eval_transform = build_transforms(model_name)

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    for model_name in MODEL_CONFIGS:
        train_loader, _, _, class_names = get_dataloaders(model_name)
        images, labels = next(iter(train_loader))
        print(f"[{model_name}] classes: {class_names}")
        print(f"[{model_name}] batch images shape: {tuple(images.shape)}")
        print(f"[{model_name}] pixel range after normalize: "
              f"{images.min():.2f} to {images.max():.2f}")

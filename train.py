"""Training loop for the plant disease classifier."""

import argparse
import time

import torch
from torch import nn, optim

from dataset import MODEL_CONFIGS, get_dataloaders
from model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
BACKBONE_LEARNING_RATE = 1e-4
DEFAULT_SEED = 0


def build_optimizer(model, model_name: str):
    if model_name == "resnet18_ft":
        # Small LR for pretrained layers so fine-tuning doesn't wipe out ImageNet features
        head_params = list(model.fc.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        return optim.Adam(
            [
                {"params": backbone_params, "lr": BACKBONE_LEARNING_RATE},
                {"params": head_params, "lr": LEARNING_RATE},
            ]
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return optim.Adam(trainable_params, lr=LEARNING_RATE)


def checkpoint_path(model_name: str, strong_augment: bool = False) -> str:
    suffix = "_aug" if strong_augment else ""
    return f"best_{model_name}{suffix}.pth"


def parse_model_arg(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), default="cnn")
    return parser


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(
    model_name: str,
    seed: int = DEFAULT_SEED,
    save_path: str | None = None,
    strong_augment: bool = False,
) -> str:
    save_path = save_path or checkpoint_path(model_name, strong_augment)
    # Seeds weight init, shuffling order, and augmentation, which all draw from torch's RNG
    torch.manual_seed(seed)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(model_name, strong_augment)

    model = build_model(model_name, num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, model_name)

    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_samples += images.size(0)

        train_loss = running_loss / running_samples
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)

        print(
            f"epoch {epoch}/{NUM_EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )

        # On an accuracy tie, lower val loss means more confident correct predictions
        is_better = val_accuracy > best_val_accuracy or (
            val_accuracy == best_val_accuracy and val_loss < best_val_loss
        )
        if is_better:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(f"  saved new best model (val_accuracy={val_accuracy:.4f}, val_loss={val_loss:.4f})")

    elapsed = time.perf_counter() - start_time
    print(
        f"training done in {elapsed:.0f}s. best epoch={best_epoch} "
        f"val_accuracy={best_val_accuracy:.4f} val_loss={best_val_loss:.4f}"
    )
    return save_path


if __name__ == "__main__":
    parser = parse_model_arg(__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--strong-augment", action="store_true")
    args = parser.parse_args()
    train(args.model, seed=args.seed, strong_augment=args.strong_augment)

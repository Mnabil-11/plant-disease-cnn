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


def checkpoint_path(model_name: str) -> str:
    return f"best_{model_name}.pth"


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


def train(model_name: str):
    train_loader, val_loader, test_loader, class_names = get_dataloaders(model_name)

    model = build_model(model_name, num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)

    best_val_accuracy = 0.0
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

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path(model_name))
            print(f"  saved new best model (val_accuracy={val_accuracy:.4f})")

    elapsed = time.perf_counter() - start_time
    print(f"training done in {elapsed:.0f}s. best val_accuracy={best_val_accuracy:.4f}")


if __name__ == "__main__":
    args = parse_model_arg(__doc__).parse_args()
    train(args.model)

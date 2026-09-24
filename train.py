"""Training loop for the plant disease CNN."""

import torch
from torch import nn, optim

from dataset import get_dataloaders
from model import PlantDiseaseCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = "best_model.pth"


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


def train():
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    model = PlantDiseaseCNN(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

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
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  saved new best model (val_accuracy={val_accuracy:.4f})")

    print(f"training done. best val_accuracy={best_val_accuracy:.4f}")


if __name__ == "__main__":
    train()

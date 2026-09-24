"""Evaluate the best checkpoint on the held-out test set."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from dataset import NORM_MEAN, NORM_STD, get_dataloaders
from model import PlantDiseaseCNN
from train import CHECKPOINT_PATH, DEVICE

RESULTS_DIR = Path(__file__).resolve().parent / "results"
NUM_EXAMPLES = 6


def collect_predictions(model, loader):
    all_images, all_labels, all_preds = [], [], []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            all_images.append(images)
            all_labels.append(labels)
            all_preds.append(outputs.argmax(dim=1).cpu())

    return torch.cat(all_images), torch.cat(all_labels), torch.cat(all_preds)


def unnormalize(image):
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1).permute(1, 2, 0)


def plot_examples(images, labels, preds, class_names, indices, title, path):
    if not indices:
        print(f"no examples for: {title}")
        return

    fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3.5), squeeze=False)
    for ax, idx in zip(axes[0], indices):
        ax.imshow(unnormalize(images[idx]))
        ax.set_title(
            f"true: {class_names[labels[idx]]}\npred: {class_names[preds[idx]]}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    _, _, test_loader, class_names = get_dataloaders()

    model = PlantDiseaseCNN(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    images, labels, preds = collect_predictions(model, test_loader)

    n_correct = (preds == labels).sum().item()
    print(f"test accuracy: {n_correct / len(labels):.4f} ({n_correct}/{len(labels)})\n")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    cm = confusion_matrix(labels, preds)
    print("confusion matrix (rows = true, columns = predicted):")
    print(cm)

    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    per_class = NUM_EXAMPLES // len(class_names)
    correct_idx = [
        idx
        for class_id in range(len(class_names))
        for idx in ((preds == labels) & (labels == class_id)).nonzero().flatten().tolist()[:per_class]
    ]
    wrong_idx = (preds != labels).nonzero().flatten().tolist()[:NUM_EXAMPLES]

    plot_examples(images, labels, preds, class_names, correct_idx,
                  "Correct predictions", RESULTS_DIR / "correct_examples.png")
    plot_examples(images, labels, preds, class_names, wrong_idx,
                  "Wrong predictions", RESULTS_DIR / "wrong_examples.png")

    print(f"\nplots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

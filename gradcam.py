"""Grad-CAM heatmaps showing which image regions drive each model's prediction."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import MODEL_CONFIGS, build_transforms
from predict import load_class_names, load_model
from train import DEVICE

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "results" / "gradcam.png"

DEFAULT_IMAGES = [
    "data/test/late_blight/0441138d*",
    "data/test/late_blight/22b19227*",
    "data/test/healthy/c85bacb0*",
    "samples/late_blight.jpg",
    "samples/early_blight.jpg",
]

# Last layer that still has a spatial layout; later layers pool it away
TARGET_LAYERS = {
    "cnn": lambda model: model.features[10],  # ReLU after the 3rd conv, 64 x 32 x 32
    "resnet18": lambda model: model.layer4,  # 512 x 7 x 7
    "resnet18_ft": lambda model: model.layer4,
}


def grad_cam(model, layer, tensor, output_size):
    captured = {}

    def save_activation(module, inputs, output):
        captured["activation"] = output
        output.register_hook(lambda grad: captured.update(gradient=grad))

    handle = layer.register_forward_hook(save_activation)
    try:
        # Frozen models have no trainable weights before the head, so gradients
        # only flow back to the target layer if the input itself requires grad
        logits = model(tensor.clone().requires_grad_(True))
    finally:
        handle.remove()

    probabilities = torch.softmax(logits, dim=1)[0]
    class_idx = probabilities.argmax().item()

    model.zero_grad()
    logits[0, class_idx].backward()

    activation = captured["activation"][0]
    weights = captured["gradient"][0].mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * activation).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=output_size, mode="bilinear", align_corners=False)

    return cam[0, 0].detach().cpu(), class_idx, probabilities[class_idx].item()


def resolve_images(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(ROOT.glob(pattern)) or [Path(pattern)]
        paths.append(matches[0])
    return paths


def true_label(path: Path, class_names) -> str:
    for candidate in (path.parent.name, path.stem):
        if candidate in class_names:
            return candidate
    return "unknown"


def main(image_patterns):
    class_names = load_class_names()
    model_names = list(MODEL_CONFIGS)
    models = {name: load_model(name, len(class_names)) for name in model_names}
    image_paths = resolve_images(image_patterns)

    n_rows, n_cols = len(image_paths), 1 + len(model_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows), squeeze=False)

    for row, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        output_size = (image.height, image.width)

        axes[row, 0].imshow(image)
        short_name = path.stem if len(path.stem) <= 15 else path.name[:8]
        axes[row, 0].set_title(f"{short_name}\ntrue: {true_label(path, class_names)}", fontsize=9)

        for col, name in enumerate(model_names, start=1):
            _, eval_transform = build_transforms(name)
            tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
            cam, class_idx, confidence = grad_cam(models[name], TARGET_LAYERS[name](models[name]), tensor, output_size)

            axes[row, col].imshow(image)
            axes[row, col].imshow(cam, cmap="jet", alpha=0.45)
            axes[row, col].set_title(f"{name}\npred: {class_names[class_idx]} ({confidence:.0%})", fontsize=9)

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=110)
    plt.close(fig)
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", default=DEFAULT_IMAGES,
                        help="image paths or glob patterns (defaults to a set of hard test images)")
    main(parser.parse_args().images)

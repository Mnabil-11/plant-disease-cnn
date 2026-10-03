"""Export the fine-tuned ResNet18 to ONNX for in-browser inference, and check it matches PyTorch."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from dataset import CLASS_NAMES, MODEL_CONFIGS, build_transforms
from predict import load_model

MODEL_NAME = "resnet18_ft"
OUTPUT_PATH = ROOT / "web" / "model.onnx"


def main():
    model = load_model(MODEL_NAME, len(CLASS_NAMES))
    size = MODEL_CONFIGS[MODEL_NAME]["image_size"]

    with torch.no_grad():
        torch.onnx.export(
            model,
            (torch.randn(1, 3, size, size),),
            OUTPUT_PATH,
            input_names=["input"],
            output_names=["logits"],
            external_data=False,
        )
    print(f"saved {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")

    session = ort.InferenceSession(str(OUTPUT_PATH))
    _, eval_transform = build_transforms(MODEL_NAME)
    for path in sorted((ROOT / "samples").glob("*.jpg")):
        x = eval_transform(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            torch_logits = model(x).numpy()
        onnx_logits = session.run(None, {"input": x.numpy()})[0]
        max_diff = np.abs(torch_logits - onnx_logits).max()
        prediction = CLASS_NAMES[int(onnx_logits.argmax())]
        print(f"{path.name:18} onnx prediction: {prediction:13} max logit diff vs torch: {max_diff:.2e}")


if __name__ == "__main__":
    main()

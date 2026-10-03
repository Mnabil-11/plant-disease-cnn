"""Quantize web/model.onnx to int8 and compare it with the float32 model on the test set."""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)
from PIL import Image

from dataset import CLASS_NAMES, DATA_DIR, build_transforms

MODEL_NAME = "resnet18_ft"
FLOAT_PATH = ROOT / "web" / "model.onnx"
INT8_PATH = ROOT / "web" / "model_int8.onnx"
CALIBRATION_IMAGES_PER_CLASS = 20


def load_input(path: Path) -> np.ndarray:
    _, eval_transform = build_transforms(MODEL_NAME)
    return eval_transform(Image.open(path).convert("RGB")).unsqueeze(0).numpy()


class TrainImageReader(CalibrationDataReader):
    # Calibrate on training images only; the test set must stay unseen until evaluation
    def __init__(self):
        paths = []
        for class_name in CLASS_NAMES:
            paths += sorted((DATA_DIR / "train" / class_name).iterdir())[:CALIBRATION_IMAGES_PER_CLASS]
        self.inputs = iter({"input": load_input(p)} for p in paths)

    def get_next(self):
        return next(self.inputs, None)


def softmax(logits: np.ndarray) -> np.ndarray:
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


def compare_on_test_set() -> None:
    sessions = {
        "float32": ort.InferenceSession(str(FLOAT_PATH)),
        "int8": ort.InferenceSession(str(INT8_PATH)),
    }
    test_items = [(p, i) for i, c in enumerate(CLASS_NAMES) for p in sorted((DATA_DIR / "test" / c).iterdir())]
    labels = np.array([label for _, label in test_items])

    probs = {}
    for name, session in sessions.items():
        logits = np.concatenate([session.run(None, {"input": load_input(p)})[0] for p, _ in test_items])
        probs[name] = softmax(logits)

    for name, path in [("float32", FLOAT_PATH), ("int8", INT8_PATH)]:
        preds = probs[name].argmax(axis=1)
        correct = (preds == labels).sum()
        print(f"{name:8} size {path.stat().st_size / 1e6:5.1f} MB | test accuracy {correct / len(labels):.4f} ({correct}/{len(labels)})")

    agree = (probs["float32"].argmax(axis=1) == probs["int8"].argmax(axis=1)).sum()
    diff = np.abs(probs["float32"] - probs["int8"])
    print(f"same prediction on {agree}/{len(labels)} test images")
    print(f"probability difference: mean {diff.mean() * 100:.2f} pp, max {diff.max() * 100:.2f} pp")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # Shape inference and graph optimizations (e.g. folding BatchNorm into Conv) before quantizing
        prepared = Path(tmp) / "prepared.onnx"
        quant_pre_process(str(FLOAT_PATH), str(prepared))
        quantize_static(
            str(prepared),
            str(INT8_PATH),
            TrainImageReader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )
    print(f"saved {INT8_PATH}")
    compare_on_test_set()


if __name__ == "__main__":
    main()

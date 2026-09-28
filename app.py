"""Gradio web demo for the plant disease classifier."""

from functools import cache
from pathlib import Path

import gradio as gr

from dataset import MODEL_CONFIGS
from predict import load_class_names, load_model, predict

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
DEFAULT_MODEL = "resnet18_ft"
CLASS_NAMES = load_class_names()

DESCRIPTION = """
Upload a photo of a single potato leaf to classify it as **healthy**, **early blight**, or **late blight**.

The models were trained on PlantVillage images, which show one leaf on a plain gray background.
Photos taken in the field (soil, other leaves, shadows) will likely be less accurate.
"""


@cache
def get_model(model_name: str):
    return load_model(model_name, len(CLASS_NAMES))


def classify(image, model_name: str):
    if image is None:
        raise gr.Error("Please upload a leaf image first.")
    if not model_name:
        raise gr.Error("Please choose a model.")
    return predict(get_model(model_name), model_name, image, CLASS_NAMES)


demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Image(type="pil", label="Leaf image"),
        gr.Dropdown(list(MODEL_CONFIGS), value=DEFAULT_MODEL, label="Model"),
    ],
    outputs=gr.Label(num_top_classes=len(CLASS_NAMES), label="Prediction"),
    examples=[[str(path), DEFAULT_MODEL] for path in sorted(SAMPLES_DIR.glob("*.jpg"))],
    title="Potato Leaf Disease Classifier",
    description=DESCRIPTION,
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()

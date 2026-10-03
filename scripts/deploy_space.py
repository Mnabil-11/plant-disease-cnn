"""Deploy the Gradio demo to a Hugging Face Space.

Requires being logged in first: `hf auth login` with a write token.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
GITHUB_URL = "https://github.com/Mnabil-11/plant-disease-cnn"

CODE_FILES = ["app.py", "predict.py", "dataset.py", "model.py", "train.py"]
CHECKPOINTS = ["best_cnn.pth", "best_resnet18.pth", "best_resnet18_ft.pth"]

# CPU-only wheels: Spaces' free hardware has no GPU, and the CUDA build is several GB
SPACE_REQUIREMENTS = """\
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.14.0
torchvision==0.29.0
"""

SPACE_README = f"""\
---
title: Potato Leaf Disease Classifier
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Classify potato leaves as healthy, early or late blight
---

Classifies a potato leaf photo as healthy, early blight, or late blight. It compares a small CNN
trained from scratch with frozen and fine-tuned ResNet18 models, all trained on a PlantVillage sample.

Code, training details, and evaluation: {GITHUB_URL}
"""


def build_bundle(target: Path) -> None:
    for name in CODE_FILES + CHECKPOINTS:
        shutil.copy2(ROOT / name, target / name)
    shutil.copytree(ROOT / "samples", target / "samples")
    (target / "requirements.txt").write_text(SPACE_REQUIREMENTS)
    (target / "README.md").write_text(SPACE_README, encoding="utf-8")


def main(repo_name: str, private: bool) -> None:
    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{repo_name}"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        build_bundle(bundle)
        print("uploading:", sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()))
        api.upload_folder(
            folder_path=bundle,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy Gradio demo",
        )

    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-name", default="plant-disease-cnn")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    main(args.repo_name, args.private)

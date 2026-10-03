"""Deploy the in-browser demo (web/) to a static Hugging Face Space.

Static Spaces are free; the model runs client-side with ONNX Runtime Web.
Requires `python scripts/export_onnx.py` first, and being logged in with `hf auth login`.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
GITHUB_URL = "https://github.com/Mnabil-11/plant-disease-cnn"

SPACE_README = f"""\
---
title: Potato Leaf Disease Classifier
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
short_description: Classify potato leaves as healthy, early or late blight
---

Classifies a potato leaf photo as healthy, early blight, or late blight with a fine-tuned ResNet18.
The model runs in your browser with ONNX Runtime Web, so images are not uploaded anywhere.

Code, training details, and evaluation: {GITHUB_URL}
"""


def build_bundle(target: Path) -> None:
    model_path = WEB_DIR / "model.onnx"
    if not model_path.exists():
        raise SystemExit("web/model.onnx not found. Run `python scripts/export_onnx.py` first.")

    shutil.copy2(WEB_DIR / "index.html", target / "index.html")
    shutil.copy2(model_path, target / "model.onnx")
    shutil.copytree(ROOT / "samples", target / "samples")
    (target / "README.md").write_text(SPACE_README, encoding="utf-8")


def main(repo_name: str, private: bool) -> None:
    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{repo_name}"
    api.create_repo(repo_id, repo_type="space", space_sdk="static", private=private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        build_bundle(bundle)
        print("uploading:", sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()))
        api.upload_folder(
            folder_path=bundle,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy in-browser demo",
        )

    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-name", default="plant-disease-cnn")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    main(args.repo_name, args.private)

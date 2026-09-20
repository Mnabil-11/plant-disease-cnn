"""Download a small, balanced sample of the PlantVillage dataset.

Source: https://github.com/spMohanty/PlantVillage-Dataset (raw/color)

We only take ~150 images per class so the whole project stays fast to
train and easy to inspect while learning.
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "spMohanty/PlantVillage-Dataset"
BRANCH = "master"
IMAGES_PER_CLASS = 150

# local_folder_name -> folder name inside the GitHub repo
CLASSES = {
    "healthy": "Potato___healthy",
    "early_blight": "Potato___Early_blight",
    "late_blight": "Potato___Late_blight",
}

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def list_files(remote_folder: str) -> list[str]:
    url = f"https://api.github.com/repos/{REPO}/contents/raw/color/{remote_folder}"
    with urllib.request.urlopen(url) as response:
        entries = json.load(response)
    return [entry["name"] for entry in entries]


def download_file(remote_folder: str, filename: str, dest_path: Path) -> None:
    url = (
        f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/raw/color/"
        f"{remote_folder}/{urllib.parse.quote(filename)}"
    )
    urllib.request.urlretrieve(url, dest_path)


def main() -> None:
    for local_name, remote_folder in CLASSES.items():
        class_dir = OUT_DIR / local_name
        class_dir.mkdir(parents=True, exist_ok=True)

        filenames = list_files(remote_folder)[:IMAGES_PER_CLASS]
        print(f"{local_name}: downloading {len(filenames)} images...")

        for i, filename in enumerate(filenames, start=1):
            dest_path = class_dir / filename
            if dest_path.exists():
                continue
            download_file(remote_folder, filename, dest_path)
            if i % 50 == 0:
                print(f"  {i}/{len(filenames)}")

        print(f"{local_name}: done ({len(filenames)} images)")


if __name__ == "__main__":
    main()

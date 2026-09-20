"""Split data/raw/<class>/ into data/train, data/val, data/test.

Split ratio: 70% train / 15% val / 15% test, done separately per class
so each split stays balanced across classes.
"""

import random
import shutil
from pathlib import Path

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# whatever remains goes to test

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


def main() -> None:
    random.seed(RANDOM_SEED)

    class_names = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())

    for class_name in class_names:
        files = sorted((RAW_DIR / class_name).iterdir())
        random.shuffle(files)

        n_train = int(len(files) * TRAIN_RATIO)
        n_val = int(len(files) * VAL_RATIO)

        splits = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }

        for split_name, split_files in splits.items():
            split_dir = DATA_DIR / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for src_path in split_files:
                shutil.copy2(src_path, split_dir / src_path.name)

        print(
            f"{class_name}: train={len(splits['train'])} "
            f"val={len(splits['val'])} test={len(splits['test'])}"
        )


if __name__ == "__main__":
    main()

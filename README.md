# Plant Disease Classification with a CNN

A PyTorch project that classifies potato leaf images into three classes:

- `healthy`
- `early_blight`
- `late_blight`

This is an end-to-end learning project covering data preparation, model design, training, evaluation, and inference. It compares two models:

- **`cnn`**: a small CNN built from scratch (three convolutional blocks), so every layer and tensor shape can be followed and understood.
- **`resnet18`**: transfer learning from an ImageNet-pretrained ResNet18. The backbone is frozen and only a new final layer is trained.

## Results

Both models were evaluated on the same held-out test set: 69 images that were not used for training or for picking the best checkpoint.

| Metric                       | `cnn` (from scratch) | `resnet18` (frozen, pretrained) |
|------------------------------|----------------------|---------------------------------|
| Test accuracy                | 95.7% (66/69)        | 95.7% (66/69)                   |
| `late_blight` recall         | 0.870                | **0.957**                       |
| `healthy` recall             | **1.000**            | 0.913                           |
| Best validation accuracy     | 98.5% (epoch 9)      | 98.5% (epoch 8)                 |
| Trainable parameters         | 2,121,475            | 1,539                           |

**Accuracy is the same, but the errors are different:**

- `cnn`: all 3 errors are **sick leaves predicted as `healthy`**. These are missed infections, which is the costliest mistake for this task.
- `resnet18`: 2 of its 3 errors are **healthy leaves predicted as `late_blight`** (false alarms). Only 1 infection is missed.

For disease detection a false alarm is cheaper than a missed infection, so `resnet18` behaves better in practice even though the accuracy is identical. Note that the gap is only 2 or 3 images out of 69, so this is a promising trend rather than a conclusive result.

### `cnn`

![Confusion matrix, cnn](results/cnn/confusion_matrix.png)

The misclassified `late_blight` leaves are mostly green, with one or two small lesions, which suggests early-stage infection:

![Wrong predictions, cnn](results/cnn/wrong_examples.png)

### `resnet18`

![Confusion matrix, resnet18](results/resnet18/confusion_matrix.png)

The two false alarms are healthy leaves with a bright, rough surface texture:

![Wrong predictions, resnet18](results/resnet18/wrong_examples.png)

Correct predictions from each model are saved in `results/<model>/correct_examples.png`.

## Dataset

A small, balanced sample of the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset) (color images, 256×256):

| Class        | Source folder           | Images |
|--------------|-------------------------|--------|
| healthy      | `Potato___healthy`      | 150    |
| early_blight | `Potato___Early_blight` | 150    |
| late_blight  | `Potato___Late_blight`  | 150    |

Each class is split separately into 70% train, 15% validation, and 15% test (105 / 22 / 23 images per class). The split uses a fixed random seed, so it is reproducible.

## Models

### `cnn` (from scratch)

```
Input: 3 × 128 × 128
Conv(3→16)  + BatchNorm + ReLU + MaxPool  →  16 × 64 × 64
Conv(16→32) + BatchNorm + ReLU + MaxPool  →  32 × 32 × 32
Conv(32→64) + BatchNorm + ReLU + MaxPool  →  64 × 16 × 16
Flatten → Linear(16384 → 128) + ReLU + Dropout(0.3)
Linear(128 → 3)
```

Inputs are normalized with a mean and std of 0.5 per channel.

### `resnet18` (transfer learning)

- ResNet18 with ImageNet weights (`IMAGENET1K_V1`) from torchvision.
- All pretrained layers are frozen (`requires_grad = False`).
- The final 1000-class layer is replaced by `Linear(512 → 3)`, which is the only part that is trained.
- Inputs are resized to 224×224 and normalized with the ImageNet mean and std, matching the preprocessing used during pretraining.

### Training setup (both models)

- Loss: `CrossEntropyLoss`
- Optimizer: Adam, learning rate 1e-3, applied only to trainable parameters
- 15 epochs, batch size 32
- Augmentation (training set only): random horizontal flip and random rotation of up to 15°
- The checkpoint with the best validation accuracy is kept.

`resnet18` trained in about 3 minutes on CPU. Its validation loss was still decreasing at epoch 15, so more epochs may help.

Training is not seeded, so a new run will produce slightly different numbers.

## Project structure

```
├── scripts/
│   ├── download_data.py   # download the PlantVillage sample into data/raw/
│   └── split_data.py      # split data/raw/ into data/train, data/val, data/test
├── dataset.py             # per-model transforms and DataLoaders
├── model.py               # small CNN and pretrained ResNet18
├── train.py               # training loop, saves best_<model>.pth
├── evaluate.py            # test-set metrics and plots, saved to results/<model>/
├── predict.py             # predict the class of a single image
├── samples/               # one unseen image per class for quick testing
├── results/
│   ├── cnn/
│   └── resnet18/
└── requirements.txt
```

`data/` and the `best_<model>.pth` checkpoints are not tracked in git. Running the steps below regenerates them.

## How to run

Requires Python 3.11. The commands below are for Windows; on macOS or Linux, activate the environment with `source .venv/bin/activate`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py   # about 450 images
python scripts/split_data.py
```

Every script accepts `--model cnn` (the default) or `--model resnet18`:

```bash
python train.py --model resnet18
python evaluate.py --model resnet18
python predict.py --model resnet18 samples/late_blight.jpg
```

The first `resnet18` run downloads the pretrained weights (about 45 MB) from download.pytorch.org.

Example of `predict.py` on `samples/late_blight.jpg`, an early-stage `late_blight` leaf with a single small lesion:

```
model: resnet18
image: late_blight.jpg
prediction: late_blight (81.0% confidence)

  late_blight    81.0%
  healthy        11.1%
  early_blight   8.0%
```

On the same image, `cnn` is also correct but only 58.1% confident (41.8% `healthy`).

## Limitations

- **Small dataset.** There are only 105 training images per class, and the test set has 69 images, so each test image changes accuracy by about 1.5 percentage points. The difference between the two models is within this noise.
- **Lab conditions.** All PlantVillage images have a plain background and controlled lighting. Performance on real field photos (soil, other leaves, shadows) is expected to be noticeably worse because of domain shift.
- **Early-stage late blight.** Leaves with small lesions are the hardest case for `cnn`, and missing them is the costliest mistake.
- **Frozen BatchNorm statistics.** The frozen ResNet18 layers are not trained, but their BatchNorm running statistics still update in training mode.

## Possible improvements

- Fine-tune the whole ResNet18 with a small learning rate instead of only training the last layer.
- Train `resnet18` for more epochs, since its validation loss had not stopped decreasing.
- Collect more images, especially early-stage `late_blight` examples.
- Flag low-confidence predictions (for example below 80%) for manual review instead of returning a hard label.

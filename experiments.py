"""Train every model with several seeds and summarize test-set results."""

import argparse
import csv
import statistics
from pathlib import Path

import torch

from dataset import MODEL_CONFIGS, get_dataloaders
from evaluate import collect_predictions
from model import build_model
from train import DEVICE, train

SEEDS = [0, 1, 2]
ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"


def test_metrics(model_name: str, checkpoint: str) -> dict:
    _, _, test_loader, class_names = get_dataloaders(model_name)
    model = build_model(model_name, num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    _, labels, preds = collect_predictions(model, test_loader)

    healthy = class_names.index("healthy")
    late_blight = class_names.index("late_blight")
    is_diseased = labels != healthy

    return {
        "test_accuracy": (preds == labels).float().mean().item(),
        "late_blight_recall": (preds[labels == late_blight] == late_blight).float().mean().item(),
        "missed_infections": (is_diseased & (preds == healthy)).sum().item(),
        "false_alarms": (~is_diseased & (preds != healthy)).sum().item(),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summary_table(rows: list[dict]) -> str:
    lines = [
        "| Model | Test accuracy | `late_blight` recall | Missed infections | False alarms |",
        "|---|---|---|---|---|",
    ]
    for model_name in MODEL_CONFIGS:
        runs = [r for r in rows if r["model"] == model_name]
        acc = [r["test_accuracy"] * 100 for r in runs]
        recall = [r["late_blight_recall"] for r in runs]
        missed = [r["missed_infections"] for r in runs]
        alarms = [r["false_alarms"] for r in runs]
        lines.append(
            f"| `{model_name}` "
            f"| {statistics.mean(acc):.1f}% ± {statistics.stdev(acc):.1f} "
            f"| {statistics.mean(recall):.3f} ± {statistics.stdev(recall):.3f} "
            f"| {statistics.mean(missed):.1f} ({', '.join(map(str, missed))}) "
            f"| {statistics.mean(alarms):.1f} ({', '.join(map(str, alarms))}) |"
        )
    return "\n".join(lines)


def main(strong_augment: bool):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    suffix = "_aug" if strong_augment else ""
    results_csv = ROOT / "results" / f"experiments{suffix}.csv"
    rows = []

    for model_name in MODEL_CONFIGS:
        for seed in SEEDS:
            print(f"=== {model_name}{suffix} seed={seed} ===", flush=True)
            checkpoint = train(
                model_name,
                seed=seed,
                save_path=str(CHECKPOINT_DIR / f"{model_name}{suffix}_seed{seed}.pth"),
                strong_augment=strong_augment,
            )
            metrics = test_metrics(model_name, checkpoint)
            rows.append({"model": model_name, "seed": seed, **metrics})
            write_csv(rows, results_csv)
            print(f"result: {metrics}", flush=True)

    print(f"\nseeds: {SEEDS}, strong_augment: {strong_augment}\n")
    print(summary_table(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-augment", action="store_true")
    main(parser.parse_args().strong_augment)

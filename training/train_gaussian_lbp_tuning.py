"""Train and compare YOLO models for generated Gaussian + LBP datasets."""

import csv
import json
from pathlib import Path
import time

import torch
from ultralytics import YOLO


# =============================================================================
# EDIT THESE SETTINGS BEFORE RUNNING
# =============================================================================

PROCESSED_ROOT = Path(
    r"C:\Users\ASUS\Downloads\pcb_dataset_gaussian_lbp"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_BASE = PROJECT_ROOT / "runs" / "gaussian_lbp_tuning"
RUN_LABEL = "screening"

MODEL_NAME = "yolov8n.pt"

# Use 25 for screening. Increase this only for the best configurations.
EPOCHS = 25
IMAGE_SIZE = 640
TRAIN_BATCH_SIZE = -1
VAL_BATCH_SIZE = 4
SEED = 42
PATIENCE = 10
WORKERS = 0  # Safe setting for Windows.

# Changing EPOCHS or SEED automatically creates a separate results directory.
RUNS_ROOT = RUNS_BASE / f"{RUN_LABEL}_{EPOCHS}epochs_seed{SEED}"

# None automatically selects CUDA GPU 0 when available, otherwise CPU.
# Set DEVICE_OVERRIDE to 0, 1, or "cpu" when a specific device is required.
DEVICE_OVERRIDE = None

# Leave empty to train every generated directory containing data.yaml.
# Example: SELECTED_CONFIGURATIONS = ["g3_s05_p8_r1", "g5_s10_p16_r2"]
SELECTED_CONFIGURATIONS = []


# =============================================================================
# DISCOVERY AND VALIDATION
# =============================================================================

def get_device():
    if DEVICE_OVERRIDE is not None:
        return DEVICE_OVERRIDE
    return 0 if torch.cuda.is_available() else "cpu"


def discover_datasets():
    if not PROCESSED_ROOT.is_dir():
        raise FileNotFoundError(
            f"Processed dataset root not found: {PROCESSED_ROOT}\n"
            "Run generate_gaussian_lbp_datasets.py first."
        )

    datasets = []
    for dataset_dir in sorted(PROCESSED_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if SELECTED_CONFIGURATIONS and (
            dataset_dir.name not in SELECTED_CONFIGURATIONS
        ):
            continue

        yaml_path = dataset_dir / "data.yaml"
        if yaml_path.is_file():
            datasets.append(dataset_dir)

    if not datasets:
        raise RuntimeError(
            "No generated dataset directories containing data.yaml were found."
        )

    if SELECTED_CONFIGURATIONS:
        found = {path.name for path in datasets}
        missing = set(SELECTED_CONFIGURATIONS) - found
        if missing:
            raise FileNotFoundError(
                f"Selected configurations not found: {sorted(missing)}"
            )

    return datasets


def read_parameters(dataset_dir):
    manifest_path = dataset_dir / "preprocessing.json"
    if not manifest_path.is_file():
        return {
            "kernel_size": "",
            "sigma": "",
            "lbp_points": "",
            "lbp_radius": "",
            "lbp_method": "",
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "kernel_size": manifest["gaussian"]["kernel_size"],
        "sigma": manifest["gaussian"]["sigma"],
        "lbp_points": manifest["lbp"]["points"],
        "lbp_radius": manifest["lbp"]["radius"],
        "lbp_method": manifest["lbp"]["method"],
    }


# =============================================================================
# TRAINING AND VALIDATION
# =============================================================================

def train_configuration(dataset_dir, device):
    name = dataset_dir.name
    yaml_path = dataset_dir / "data.yaml"

    print("\n" + "=" * 80)
    print(f"TRAINING {name}")
    print("=" * 80)

    # A fresh pretrained model makes every comparison start identically.
    model = YOLO(MODEL_NAME)
    train_batch_size = (
        16
        if device == "cpu" and TRAIN_BATCH_SIZE == -1
        else TRAIN_BATCH_SIZE
    )
    start_time = time.time()

    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=train_batch_size,
        device=device,
        seed=SEED,
        deterministic=True,
        patience=PATIENCE,
        workers=WORKERS,
        project=str(RUNS_ROOT),
        name=name,
        exist_ok=True,
        plots=True,
        # The generated channels are not natural HSV colour channels.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )

    training_time = time.time() - start_time
    best_model = Path(model.trainer.best)

    if not best_model.is_file():
        raise FileNotFoundError(f"Best model was not created: {best_model}")

    return yaml_path, best_model, training_time


def validate_configuration(
    dataset_dir,
    yaml_path,
    best_model,
    training_time,
    device,
):
    name = dataset_dir.name

    print("\n" + "-" * 80)
    print(f"VALIDATING {name}")
    print("-" * 80)

    model = YOLO(str(best_model))
    metrics = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=VAL_BATCH_SIZE,
        device=device,
        workers=WORKERS,
        project=str(RUNS_ROOT),
        name=f"{name}_validation",
        exist_ok=True,
        plots=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    parameters = read_parameters(dataset_dir)
    result = {
        "configuration": name,
        **parameters,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "training_time_seconds": training_time,
        "best_model": str(best_model.resolve()),
    }

    per_class = []
    names = model.names
    for class_id, class_map in enumerate(metrics.box.maps):
        class_name = (
            names.get(class_id, str(class_id))
            if isinstance(names, dict)
            else names[class_id]
        )
        per_class.append(
            {
                "configuration": name,
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(metrics.box.p[class_id]),
                "recall": float(metrics.box.r[class_id]),
                "map50": float(metrics.box.ap50[class_id]),
                "map50_95": float(class_map),
            }
        )

    print(
        f"Precision={precision:.4f}, Recall={recall:.4f}, "
        f"mAP50={map50:.4f}, mAP50-95={map50_95:.4f}"
    )

    return result, per_class


# =============================================================================
# RESULT REPORTING
# =============================================================================

SUMMARY_FIELDS = [
    "configuration",
    "kernel_size",
    "sigma",
    "lbp_points",
    "lbp_radius",
    "lbp_method",
    "precision",
    "recall",
    "f1",
    "map50",
    "map50_95",
    "training_time_seconds",
    "best_model",
]

PER_CLASS_FIELDS = [
    "configuration",
    "class_id",
    "class_name",
    "precision",
    "recall",
    "map50",
    "map50_95",
]


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results(results, per_class_results):
    ranked = sorted(
        results,
        key=lambda result: result["map50_95"],
        reverse=True,
    )

    save_csv(
        RUNS_ROOT / "parameter_tuning_results.csv",
        ranked,
        SUMMARY_FIELDS,
    )
    save_csv(
        RUNS_ROOT / "parameter_tuning_per_class.csv",
        per_class_results,
        PER_CLASS_FIELDS,
    )


def print_comparison(results):
    ranked = sorted(
        results,
        key=lambda result: result["map50_95"],
        reverse=True,
    )

    print("\n" + "=" * 100)
    print("GAUSSIAN + LBP VALIDATION RANKING")
    print("=" * 100)
    print(
        f"{'Rank':<7}{'Configuration':<24}{'Precision':<12}"
        f"{'Recall':<12}{'mAP50':<12}{'mAP50-95':<12}"
    )

    for rank, result in enumerate(ranked, start=1):
        print(
            f"{rank:<7}{result['configuration']:<24}"
            f"{result['precision']:<12.4f}{result['recall']:<12.4f}"
            f"{result['map50']:<12.4f}{result['map50_95']:<12.4f}"
        )

    best = ranked[0]
    print("\nBest tested configuration:")
    print(f"  Name       : {best['configuration']}")
    print(f"  Gaussian   : {best['kernel_size']}x{best['kernel_size']}")
    print(f"  Sigma      : {best['sigma']}")
    print(f"  LBP        : P={best['lbp_points']}, R={best['lbp_radius']}")
    print(f"  mAP50-95   : {best['map50_95']:.4f}")
    print(f"  Best model : {best['best_model']}")


def main():
    datasets = discover_datasets()
    device = get_device()

    print("=" * 80)
    print("GAUSSIAN + LBP PARAMETER TUNING")
    print("=" * 80)
    print(f"Datasets: {len(datasets)}")
    print(f"Epochs: {EPOCHS}")
    print(f"Device: {device}")

    results = []
    per_class_results = []

    for dataset_dir in datasets:
        yaml_path, best_model, training_time = train_configuration(
            dataset_dir,
            device,
        )
        result, class_results = validate_configuration(
            dataset_dir,
            yaml_path,
            best_model,
            training_time,
            device,
        )

        results.append(result)
        per_class_results.extend(class_results)

        # Preserve completed results if a later configuration is interrupted.
        save_results(results, per_class_results)

    print_comparison(results)
    print(f"\nResults saved under: {RUNS_ROOT}")


if __name__ == "__main__":
    main()

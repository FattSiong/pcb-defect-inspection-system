"""Train and validate YOLOv8n on the bilateral-only B1 dataset."""

from pathlib import Path
import csv
import json
import time

import torch
from ultralytics import YOLO


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

DATASET_ROOT = Path(
    r"C:\Users\hangg\Downloads\pcb_dataset_bilateral_only_b1"
)
DATA_YAML = DATASET_ROOT / "data.yaml"
PROCESSING_CONFIG = DATASET_ROOT / "processing_config.json"

RUNS_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "bilateral_only"
    / "seed_42"
)
EXPERIMENT_NAME = "bilateral_only_b1"


# =========================================================
# LOCKED BILATERAL-ONLY CONFIGURATION
# =========================================================

EXPECTED_PROCESSING = {
    "bilateral_d": 3,
    "sigma_color": 15.0,
    "sigma_space": 3.0,
    "kernel_sizes": [7, 15, 31],
    "alpha": 0.0,
    "beta": 0.0,
}

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}


# =========================================================
# FIXED YOLO SETTINGS
# =========================================================

EPOCHS = 100
IMAGE_SIZE = 640
TRAIN_BATCH_SIZE = -1
VAL_BATCH_SIZE = 4
DEVICE = 0
SEED = 42
WORKERS = 4
PATIENCE = 150

IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


# =========================================================
# DATASET AND ENVIRONMENT CHECKS
# =========================================================

def count_split(split):
    image_directory = DATASET_ROOT / split / "images"
    label_directory = DATASET_ROOT / split / "labels"

    if not image_directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_directory}"
        )

    if not label_directory.exists():
        raise FileNotFoundError(
            f"Label directory not found: {label_directory}"
        )

    images = {
        path.stem
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    labels = {
        path.stem
        for path in label_directory.glob("*.txt")
    }

    if not images:
        raise RuntimeError(f"No {split} images were found.")

    if images != labels:
        raise RuntimeError(
            f"{split} images and labels do not match. "
            f"Images without labels: {len(images - labels)}; "
            f"labels without images: {len(labels - images)}."
        )

    return len(images)


def read_processing_configuration():
    if not PROCESSING_CONFIG.exists():
        raise FileNotFoundError(
            f"Processing configuration not found: {PROCESSING_CONFIG}"
        )

    configuration = json.loads(
        PROCESSING_CONFIG.read_text(encoding="utf-8")
    )
    bilateral = configuration.get("bilateral_filter", {})
    morphology = configuration.get("morphology", {})

    actual = {
        "bilateral_d": bilateral.get("diameter"),
        "sigma_color": bilateral.get("sigma_color"),
        "sigma_space": bilateral.get("sigma_space"),
        "kernel_sizes": morphology.get("kernel_sizes"),
        "alpha": morphology.get("alpha"),
        "beta": morphology.get("beta"),
    }

    if actual != EXPECTED_PROCESSING:
        raise RuntimeError(
            "The dataset does not use the expected bilateral-only B1 "
            "configuration.\n"
            f"Expected: {EXPECTED_PROCESSING}\n"
            f"Actual  : {actual}"
        )

    generated_splits = set(configuration.get("splits", []))
    required_splits = {"train", "val", "test"}

    if not required_splits.issubset(generated_splits):
        raise RuntimeError(
            "The dataset must contain train, val and test splits."
        )

    return actual


def check_environment():
    print("=" * 80)
    print("BILATERAL-ONLY B1 YOLO TRAINING")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLOv8n checkpoint not found: {MODEL_PATH}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    processing = read_processing_configuration()
    split_counts = {
        split: count_split(split)
        for split in ("train", "val", "test")
    }

    print(f"GPU             : {torch.cuda.get_device_name(DEVICE)}")
    print(f"Model           : {MODEL_PATH}")
    print(f"Dataset         : {DATA_YAML}")
    print(f"Train images    : {split_counts['train']}")
    print(f"Validation images: {split_counts['val']}")
    print(f"Test images     : {split_counts['test']} (not used now)")
    print(f"Epochs          : {EPOCHS}")
    print(f"Image size      : {IMAGE_SIZE}")
    print(f"Training batch  : {TRAIN_BATCH_SIZE}")
    print(f"Validation batch: {VAL_BATCH_SIZE}")
    print(f"Device          : {DEVICE}")
    print(f"Seed            : {SEED}")
    print(f"Workers         : {WORKERS}")
    print(f"Patience        : {PATIENCE}")
    print(f"Processing      : {processing}")
    print(f"Output          : {RUNS_ROOT}")


# =========================================================
# TRAINING
# =========================================================

def train_model():
    print("\n" + "=" * 80)
    print("TRAINING A FRESH YOLOv8n MODEL")
    print("=" * 80)

    # Do not reuse the final hybrid model. Start every method from
    # the same original YOLOv8n pretrained checkpoint.
    model = YOLO(str(MODEL_PATH))
    start_time = time.perf_counter()

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=TRAIN_BATCH_SIZE,
        device=DEVICE,
        seed=SEED,
        deterministic=True,
        workers=WORKERS,
        patience=PATIENCE,
        project=str(RUNS_ROOT),
        name=f"{EXPERIMENT_NAME}_train",
        exist_ok=False,
        save=True,
        plots=True,
        val=True,
        cache=False,
    )

    training_seconds = time.perf_counter() - start_time
    best_model = Path(model.trainer.best)

    if not best_model.exists():
        raise FileNotFoundError(
            f"best.pt was not created: {best_model}"
        )

    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Best model   : {best_model}")

    return best_model, training_seconds


# =========================================================
# VALIDATION (TEST REMAINS UNTOUCHED)
# =========================================================

def validate_model(best_model):
    print("\n" + "=" * 80)
    print("VALIDATING BEST.PT ON THE VALIDATION SPLIT")
    print("=" * 80)

    model = YOLO(str(best_model))
    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=VAL_BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(RUNS_ROOT),
        name=f"{EXPERIMENT_NAME}_validation",
        exist_ok=False,
        plots=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    per_class_maps = list(metrics.box.maps)
    result = {
        "configuration": "bilateral_only_b1",
        "dataset": str(DATA_YAML),
        "seed": SEED,
        **EXPECTED_PROCESSING,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "map50": map50,
        "map50_95": map50_95,
    }
    class_values = []

    for class_id, class_name in CLASS_NAMES.items():
        class_map = (
            float(per_class_maps[class_id])
            if class_id < len(per_class_maps)
            else 0.0
        )
        result[f"{class_name}_map50_95"] = class_map
        class_values.append(class_map)

    result["lowest_class_map50_95"] = min(class_values)

    print(f"Precision       : {precision:.6f}")
    print(f"Recall          : {recall:.6f}")
    print(f"F1-score        : {f1_score:.6f}")
    print(f"mAP50           : {map50:.6f}")
    print(f"mAP50-95        : {map50_95:.6f}")
    print(
        "Lowest class AP : "
        f"{result['lowest_class_map50_95']:.6f}"
    )

    return result


# =========================================================
# SAVE SUMMARY
# =========================================================

def save_result(result, best_model, training_seconds):
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    complete_result = {
        **result,
        "training_seconds": training_seconds,
        "best_model": str(best_model),
    }

    csv_path = RUNS_ROOT / "bilateral_only_validation_results.csv"
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(complete_result.keys()),
        )
        writer.writeheader()
        writer.writerow(complete_result)

    json_path = RUNS_ROOT / "bilateral_only_validation_results.json"
    json_path.write_text(
        json.dumps(complete_result, indent=4),
        encoding="utf-8",
    )

    print("\nResults saved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    check_environment()
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    best_model, training_seconds = train_model()
    result = validate_model(best_model)
    save_result(result, best_model, training_seconds)

    print("\n" + "=" * 80)
    print("BILATERAL-ONLY TRAINING AND VALIDATION COMPLETED")
    print("=" * 80)
    print("The test split has not been evaluated.")
    print(f"Best model:\n{best_model}")


if __name__ == "__main__":
    main()

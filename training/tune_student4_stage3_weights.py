from pathlib import Path
import csv
import json
import math
import time

import torch
from ultralytics import YOLO


# =========================================================
# STAGE 3 DATASETS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

STAGE3_DATASETS = {
    "W1": Path(
        r"C:\Users\hangg\Downloads\pcb_dataset_stage3_w1"
    ),
    # W2 reuses the winning Stage 2 M3 dataset.
    "W2": Path(
        r"C:\Users\hangg\Downloads\pcb_dataset_stage2_m3"
    ),
    "W3": Path(
        r"C:\Users\hangg\Downloads\pcb_dataset_stage3_w3"
    ),
}

# Keep all three for the first run. For repeated-seed
# confirmation, change this tuple to contain only the top two.
ACTIVE_CONFIGURATIONS = (
    "W1",
    "W2",
    "W3",
)


# =========================================================
# EXPECTED IMAGE-PROCESSING PARAMETERS
# =========================================================

EXPECTED_PARAMETERS = {
    "W1": {
        "bilateral_d": 3,
        "sigma_color": 15,
        "sigma_space": 3,
        "kernel_sizes": [7, 15, 31],
        "alpha": 0.5,
        "beta": 0.5,
    },
    "W2": {
        "bilateral_d": 3,
        "sigma_color": 15,
        "sigma_space": 3,
        "kernel_sizes": [7, 15, 31],
        "alpha": 0.7,
        "beta": 0.7,
    },
    "W3": {
        "bilateral_d": 3,
        "sigma_color": 15,
        "sigma_space": 3,
        "kernel_sizes": [7, 15, 31],
        "alpha": 1.0,
        "beta": 1.0,
    },
}


# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}


# =========================================================
# IDENTICAL TRAINING SETTINGS
# =========================================================

EPOCHS = 100
IMAGE_SIZE = 640
TRAIN_BATCH_SIZE = -1
VAL_BATCH_SIZE = 4
DEVICE = 0
SEED = 42
WORKERS = 4
PATIENCE = 150

RUNS_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "bilateral_morphology_tuning"
    / "stage3"
    / f"seed_{SEED}"
)


# =========================================================
# SELECT AND CHECK DATASETS
# =========================================================

def get_active_datasets():
    unknown = set(ACTIVE_CONFIGURATIONS) - set(STAGE3_DATASETS)

    if unknown:
        raise ValueError(
            f"Unknown configurations: {sorted(unknown)}"
        )

    return {
        name: STAGE3_DATASETS[name]
        for name in ACTIVE_CONFIGURATIONS
    }


def count_files(directory, suffix=None):
    if not directory.exists():
        return 0

    return sum(
        1
        for path in directory.iterdir()
        if (
            path.is_file()
            and (suffix is None or path.suffix.lower() == suffix)
        )
    )


def check_environment(datasets):
    print("=" * 80)
    print("STAGE 3 ENHANCEMENT WEIGHT TUNING")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the PyTorch and "
            "NVIDIA installation."
        )

    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Epochs         : {EPOCHS}")
    print(f"Image size     : {IMAGE_SIZE}")
    print(f"Training batch : {TRAIN_BATCH_SIZE}")
    print(f"Validation batch: {VAL_BATCH_SIZE}")
    print(f"Device         : {DEVICE}")
    print(f"Seed           : {SEED}")
    print(f"Workers        : {WORKERS}")
    print(f"Patience       : {PATIENCE}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pretrained model not found:\n{MODEL_PATH}"
        )

    for name, dataset_root in datasets.items():
        yaml_path = dataset_root / "data.yaml"
        config_path = dataset_root / "processing_config.json"

        if not yaml_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"{name} is incomplete. Expected:\n"
                f"{yaml_path}\n{config_path}\n"
                "Allow dataset generation to finish before training."
            )

        train_images = count_files(dataset_root / "train" / "images")
        train_labels = count_files(
            dataset_root / "train" / "labels",
            ".txt",
        )
        val_images = count_files(dataset_root / "val" / "images")
        val_labels = count_files(
            dataset_root / "val" / "labels",
            ".txt",
        )

        if train_images != 1200 or train_labels != 1200:
            raise RuntimeError(
                f"{name} train split is incomplete: "
                f"images={train_images}, labels={train_labels}."
            )

        if val_images != 1066 or val_labels != 1066:
            raise RuntimeError(
                f"{name} val split is incomplete: "
                f"images={val_images}, labels={val_labels}."
            )

        print(
            f"{name}: {dataset_root} "
            f"(train={train_images}, val={val_images})"
        )


# =========================================================
# READ AND VERIFY PREPROCESSING PARAMETERS
# =========================================================

def read_processing_configuration(name, dataset_root):
    config_path = dataset_root / "processing_config.json"

    with config_path.open("r", encoding="utf-8") as file:
        configuration = json.load(file)

    bilateral = configuration["bilateral_filter"]
    morphology = configuration["morphology"]

    actual = {
        "bilateral_d": bilateral["diameter"],
        "sigma_color": bilateral["sigma_color"],
        "sigma_space": bilateral["sigma_space"],
        "kernel_sizes": list(morphology["kernel_sizes"]),
        "alpha": morphology["alpha"],
        "beta": morphology["beta"],
    }

    expected = EXPECTED_PARAMETERS[name]

    for key, expected_value in expected.items():
        actual_value = actual[key]

        if isinstance(expected_value, float):
            matches = math.isclose(
                float(actual_value),
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        else:
            matches = actual_value == expected_value

        if not matches:
            raise ValueError(
                f"{name} has incorrect {key}: "
                f"expected {expected_value}, got {actual_value}."
            )

    processing_statistics = configuration.get(
        "processing_statistics",
        [],
    )
    total_images = sum(
        item.get("images", 0)
        for item in processing_statistics
    )
    total_time = sum(
        item.get("elapsed_seconds", 0.0)
        for item in processing_statistics
    )
    seconds_per_image = (
        total_time / total_images
        if total_images > 0
        else 0.0
    )

    return {
        **actual,
        "kernel_sizes": str(actual["kernel_sizes"]),
        "processing_seconds_per_image": seconds_per_image,
    }


# =========================================================
# TRAIN ONE CONFIGURATION
# =========================================================

def train_configuration(name, dataset_root):
    data_yaml = dataset_root / "data.yaml"

    print()
    print("=" * 80)
    print(f"TRAINING: {name}")
    print(f"Dataset : {data_yaml}")
    print("=" * 80)

    # Every configuration starts from the same fresh model.
    model = YOLO(str(MODEL_PATH))
    start_time = time.perf_counter()

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=TRAIN_BATCH_SIZE,
        device=DEVICE,
        seed=SEED,
        deterministic=True,
        workers=WORKERS,
        patience=PATIENCE,
        project=str(RUNS_ROOT),
        name=f"{name}_train",
        exist_ok=False,
        save=True,
        plots=True,
        val=True,
        cache=False,
    )

    training_seconds = time.perf_counter() - start_time
    best_model_path = Path(model.trainer.best)

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"best.pt was not created:\n{best_model_path}"
        )

    print(f"Best model: {best_model_path}")

    return best_model_path, training_seconds


# =========================================================
# VALIDATE BEST.PT ON VAL ONLY
# =========================================================

def validate_configuration(name, dataset_root, best_model_path):
    data_yaml = dataset_root / "data.yaml"

    print()
    print("-" * 80)
    print(f"VALIDATING: {name}")
    print("-" * 80)

    model = YOLO(str(best_model_path))

    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=VAL_BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(RUNS_ROOT),
        name=f"{name}_validation",
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
    results = {
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
        results[f"{class_name}_map50_95"] = class_map
        class_values.append(class_map)

    results["lowest_class_map50_95"] = min(class_values)

    print(f"Precision       : {precision:.6f}")
    print(f"Recall          : {recall:.6f}")
    print(f"F1-score        : {f1_score:.6f}")
    print(f"mAP50           : {map50:.6f}")
    print(f"mAP50-95        : {map50_95:.6f}")
    print(
        f"Lowest class AP : "
        f"{results['lowest_class_map50_95']:.6f}"
    )

    return results


# =========================================================
# SAVE RESULTS AND WINNER
# =========================================================

def save_results(results):
    if not results:
        return

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = RUNS_ROOT / "stage3_results.csv"

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Current results saved to:\n{output_path}")


def print_and_save_comparison(results):
    ranked = sorted(
        results,
        key=lambda item: (
            item["map50_95"],
            item["lowest_class_map50_95"],
            item["f1_score"],
            -item["processing_seconds_per_image"],
        ),
        reverse=True,
    )

    print()
    print("=" * 95)
    print("STAGE 3 ENHANCEMENT WEIGHT COMPARISON")
    print("=" * 95)
    print(
        f"{'Config':<10}"
        f"{'Alpha':<10}"
        f"{'Beta':<10}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'mAP50':<12}"
        f"{'mAP50-95':<12}"
        f"{'Lowest AP':<12}"
    )
    print("-" * 95)

    for result in ranked:
        print(
            f"{result['configuration']:<10}"
            f"{result['alpha']:<10}"
            f"{result['beta']:<10}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1_score']:<12.4f}"
            f"{result['map50']:<12.4f}"
            f"{result['map50_95']:<12.4f}"
            f"{result['lowest_class_map50_95']:<12.4f}"
        )

    best = ranked[0]

    print()
    print("=" * 80)
    print("BEST STAGE 3 CONFIGURATION")
    print("=" * 80)
    print(f"Configuration : {best['configuration']}")
    print(f"Alpha         : {best['alpha']}")
    print(f"Beta          : {best['beta']}")
    print(f"mAP50-95      : {best['map50_95']:.6f}")
    print(f"F1-score      : {best['f1_score']:.6f}")
    print(
        f"Lowest class AP: "
        f"{best['lowest_class_map50_95']:.6f}"
    )

    winner_path = RUNS_ROOT / "stage3_winner.json"
    winner_path.write_text(
        json.dumps(best, indent=4),
        encoding="utf-8",
    )
    print(f"Winner saved to:\n{winner_path}")

    if len(ranked) >= 2:
        difference = ranked[0]["map50_95"] - ranked[1]["map50_95"]
        print(f"Difference from second place: {difference:.6f}")

        if difference < 0.005:
            print(
                "The top two configurations are very close. "
                "Repeat only those two with Seeds 43 and 44."
            )

    return best


# =========================================================
# MAIN
# =========================================================

def main():
    datasets = get_active_datasets()
    check_environment(datasets)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    all_results = []

    for name, dataset_root in datasets.items():
        preprocessing_parameters = read_processing_configuration(
            name,
            dataset_root,
        )

        best_model_path, training_seconds = train_configuration(
            name,
            dataset_root,
        )

        validation_results = validate_configuration(
            name,
            dataset_root,
            best_model_path,
        )

        result = {
            "configuration": name,
            "dataset": str(dataset_root),
            "seed": SEED,
            **preprocessing_parameters,
            **validation_results,
            "training_seconds": training_seconds,
            "best_model": str(best_model_path),
        }

        all_results.append(result)
        save_results(all_results)

    print_and_save_comparison(all_results)


if __name__ == "__main__":
    main()

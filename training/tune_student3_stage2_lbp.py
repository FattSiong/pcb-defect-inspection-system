"""Stage 2 tuning: compare single-scale LBP settings for YOLO."""

from pathlib import Path
import csv
import json
import math
import time

import torch
from ultralytics import YOLO


# =========================================================
# PATHS AND STAGE 2 CONFIGURATIONS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

# Change this path if the generated datasets are elsewhere.
DATASET_PARENT = Path(
    r"C:\Users\ASUS\Downloads"
)

# L1 reuses the G2 dataset because G2 already uses P=8 and R=1.
STAGE2_CONFIGURATIONS = {
    "L1": {
        "dataset": (
            DATASET_PARENT /
            "pcb_dataset_gaussian_lbp_g2"
        ),
        "lbp_points": 8,
        "lbp_radius": 1.0,
    },
    "L2": {
        "dataset": (
            DATASET_PARENT /
            "pcb_dataset_gaussian_lbp_l2"
        ),
        "lbp_points": 16,
        "lbp_radius": 2.0,
    },
    "L3": {
        "dataset": (
            DATASET_PARENT /
            "pcb_dataset_gaussian_lbp_l3"
        ),
        "lbp_points": 24,
        "lbp_radius": 3.0,
    },
}

# Keep all three for the initial Stage 2 comparison.
ACTIVE_CONFIGURATIONS = (
    "L1",
    "L2",
    "L3",
)

# Stage 1 winner and fixed fusion settings.
LOCKED_GAUSSIAN_KERNEL = 3
LOCKED_GAUSSIAN_SIGMA = 0.8
LOCKED_LBP_WEIGHT = 0.25


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
# FIXED YOLO SETTINGS
# Same settings as train_student1_blocksize.py and Stage 1.
# =========================================================

EPOCHS = 100
IMAGE_SIZE = 640
TRAIN_BATCH_SIZE = -1
VAL_BATCH_SIZE = 4
DEVICE = 0
SEED = 42
PATIENCE = 150

RUNS_ROOT = (
    PROJECT_ROOT /
    "runs" /
    "gaussian_lbp_tuning" /
    "stage2" /
    f"seed_{SEED}"
)


# =========================================================
# SELECT AND VALIDATE DATASETS
# =========================================================

def get_active_configurations():
    unknown = set(ACTIVE_CONFIGURATIONS) - set(
        STAGE2_CONFIGURATIONS
    )

    if unknown:
        raise ValueError(
            f"Unknown Stage 2 configurations: {sorted(unknown)}"
        )

    if not ACTIVE_CONFIGURATIONS:
        raise ValueError("ACTIVE_CONFIGURATIONS cannot be empty.")

    return {
        name: STAGE2_CONFIGURATIONS[name]
        for name in ACTIVE_CONFIGURATIONS
    }


def values_match(actual, expected):
    if actual is None:
        return False

    if isinstance(expected, float):
        return math.isclose(
            float(actual),
            expected,
            rel_tol=0.0,
            abs_tol=1e-9,
        )

    return actual == expected


def read_and_validate_processing_configuration(
    configuration_name,
    settings,
):
    dataset_root = settings["dataset"]
    config_path = dataset_root / "processing_config.json"

    with config_path.open("r", encoding="utf-8") as file:
        configuration = json.load(file)

    if configuration.get("method") != (
        "gaussian_uniform_lbp_luminance_fusion"
    ):
        raise RuntimeError(
            f"{configuration_name} is not a Gaussian-LBP dataset."
        )

    generated_splits = set(configuration.get("splits", []))

    if not {"train", "val"}.issubset(generated_splits):
        raise RuntimeError(
            f"{configuration_name} must contain train and val splits."
        )

    gaussian = configuration.get("gaussian_filter", {})
    lbp = configuration.get("lbp", {})
    fusion = configuration.get("fusion", {})

    expected_values = {
        "gaussian_kernel": LOCKED_GAUSSIAN_KERNEL,
        "gaussian_sigma": LOCKED_GAUSSIAN_SIGMA,
        "lbp_points": settings["lbp_points"],
        "lbp_radius": settings["lbp_radius"],
        "lbp_weight": LOCKED_LBP_WEIGHT,
        "gaussian_weight": 1.0 - LOCKED_LBP_WEIGHT,
    }
    actual_values = {
        "gaussian_kernel": gaussian.get("kernel_size"),
        "gaussian_sigma": gaussian.get("sigma"),
        "lbp_points": lbp.get("points"),
        "lbp_radius": lbp.get("radius"),
        "lbp_weight": fusion.get("lbp_weight"),
        "gaussian_weight": fusion.get("gaussian_weight"),
    }

    for parameter, expected in expected_values.items():
        actual = actual_values[parameter]

        if not values_match(actual, expected):
            raise RuntimeError(
                f"{configuration_name} {parameter} mismatch: "
                f"expected {expected}, found {actual}."
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
        "gaussian_kernel": int(
            actual_values["gaussian_kernel"]
        ),
        "gaussian_sigma": float(
            actual_values["gaussian_sigma"]
        ),
        "lbp_points": int(actual_values["lbp_points"]),
        "lbp_radius": float(actual_values["lbp_radius"]),
        "lbp_weight": float(actual_values["lbp_weight"]),
        "gaussian_weight": float(
            actual_values["gaussian_weight"]
        ),
        "processing_seconds_per_image": seconds_per_image,
    }


def check_environment(configurations):
    print("=" * 80)
    print("STAGE 2 SINGLE-SCALE LBP PARAMETER TUNING")
    print("=" * 80)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the PyTorch and NVIDIA setup."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pretrained model not found:\n{MODEL_PATH}"
        )

    print(f"GPU             : {torch.cuda.get_device_name(DEVICE)}")
    print(f"Model           : {MODEL_PATH}")
    print(f"Epochs          : {EPOCHS}")
    print(f"Image size      : {IMAGE_SIZE}")
    print(f"Training batch  : {TRAIN_BATCH_SIZE}")
    print(f"Validation batch: {VAL_BATCH_SIZE}")
    print(f"Device          : {DEVICE}")
    print(f"Seed            : {SEED}")
    print(f"Patience        : {PATIENCE}")
    print(f"Locked kernel   : {LOCKED_GAUSSIAN_KERNEL}")
    print(f"Locked sigma    : {LOCKED_GAUSSIAN_SIGMA}")
    print(f"Locked LBP weight: {LOCKED_LBP_WEIGHT}")
    print(f"Output          : {RUNS_ROOT}")

    validated_parameters = {}

    for configuration_name, settings in configurations.items():
        dataset_root = settings["dataset"]
        data_yaml = dataset_root / "data.yaml"
        config_path = dataset_root / "processing_config.json"

        if not data_yaml.exists():
            raise FileNotFoundError(
                f"{configuration_name} data.yaml not found:\n"
                f"{data_yaml}\nGenerate the dataset first."
            )

        if not config_path.exists():
            raise FileNotFoundError(
                f"{configuration_name} processing config not found:\n"
                f"{config_path}"
            )

        parameters = read_and_validate_processing_configuration(
            configuration_name,
            settings,
        )
        validated_parameters[configuration_name] = parameters

        print(
            f"{configuration_name}: "
            f"P={parameters['lbp_points']}, "
            f"R={parameters['lbp_radius']}, "
            f"dataset={dataset_root}"
        )

    return validated_parameters


# =========================================================
# TRAIN AND VALIDATE ONE CONFIGURATION
# =========================================================

def train_configuration(configuration_name, dataset_root):
    data_yaml = dataset_root / "data.yaml"

    print()
    print("=" * 80)
    print(f"TRAINING CONFIGURATION: {configuration_name}")
    print(f"Dataset: {data_yaml}")
    print("=" * 80)

    # L1 is retrained here so Stage 2 remains a self-contained comparison.
    # Every configuration starts from the same fresh pretrained checkpoint.
    model = YOLO(str(MODEL_PATH))
    start_time = time.time()

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=TRAIN_BATCH_SIZE,
        device=DEVICE,
        seed=SEED,
        patience=PATIENCE,
        project=str(RUNS_ROOT),
        name=f"{configuration_name}_train",
        exist_ok=True,
        plots=True,
    )

    training_time = time.time() - start_time
    best_model_path = Path(model.trainer.best)

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"best.pt was not created:\n{best_model_path}"
        )

    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best model   : {best_model_path}")

    return best_model_path, training_time


def validate_configuration(
    configuration_name,
    dataset_root,
    best_model_path,
):
    data_yaml = dataset_root / "data.yaml"

    print()
    print("-" * 80)
    print(f"VALIDATING CONFIGURATION: {configuration_name}")
    print("-" * 80)

    model = YOLO(str(best_model_path))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=VAL_BATCH_SIZE,
        device=DEVICE,
        project=str(RUNS_ROOT),
        name=f"{configuration_name}_validation",
        exist_ok=True,
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
        "Lowest class AP : "
        f"{results['lowest_class_map50_95']:.6f}"
    )

    return results


# =========================================================
# SAVE AND DISPLAY COMPARISON
# =========================================================

def save_results(results):
    if not results:
        return

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = RUNS_ROOT / "stage2_results.csv"

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


def print_comparison(results):
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
    print("=" * 100)
    print("STAGE 2 SINGLE-SCALE LBP PARAMETER COMPARISON")
    print("=" * 100)
    print(
        f"{'Config':<10}"
        f"{'Points':<10}"
        f"{'Radius':<10}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'mAP50':<12}"
        f"{'mAP50-95':<12}"
        f"{'Lowest AP':<12}"
    )
    print("-" * 100)

    for result in ranked:
        print(
            f"{result['configuration']:<10}"
            f"{result['lbp_points']:<10}"
            f"{result['lbp_radius']:<10.1f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1_score']:<12.4f}"
            f"{result['map50']:<12.4f}"
            f"{result['map50_95']:<12.4f}"
            f"{result['lowest_class_map50_95']:<12.4f}"
        )

    best = ranked[0]

    print()
    print("=" * 100)
    print("BEST STAGE 2 CONFIGURATION")
    print("=" * 100)
    print(f"Configuration : {best['configuration']}")
    print(f"LBP points (P): {best['lbp_points']}")
    print(f"LBP radius (R): {best['lbp_radius']}")
    print(f"mAP50-95      : {best['map50_95']:.6f}")
    print(f"mAP50         : {best['map50']:.6f}")
    print(f"F1-score      : {best['f1_score']:.6f}")
    print(
        "Lowest class AP: "
        f"{best['lowest_class_map50_95']:.6f}"
    )

    winner_path = RUNS_ROOT / "stage2_winner.json"
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
                "Consider repeating both with additional seeds."
            )

    return best


# =========================================================
# MAIN
# =========================================================

def main():
    configurations = get_active_configurations()
    validated_parameters = check_environment(configurations)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = []

    for configuration_name, settings in configurations.items():
        dataset_root = settings["dataset"]
        best_model_path, training_time = train_configuration(
            configuration_name,
            dataset_root,
        )
        validation_results = validate_configuration(
            configuration_name,
            dataset_root,
            best_model_path,
        )

        experiment_result = {
            "configuration": configuration_name,
            "dataset": str(dataset_root),
            "seed": SEED,
            **validated_parameters[configuration_name],
            **validation_results,
            "training_seconds": training_time,
            "best_model": str(best_model_path),
        }
        all_results.append(experiment_result)

        # Preserve completed results if a later experiment fails.
        save_results(all_results)

    print_comparison(all_results)


if __name__ == "__main__":
    main()


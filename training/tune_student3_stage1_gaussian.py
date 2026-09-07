"""Stage 1 tuning: compare Gaussian settings for Gaussian-LBP YOLO."""

from pathlib import Path
import csv
import json
import math
import time

import torch
from ultralytics import YOLO


# =========================================================
# PATHS AND STAGE 1 CONFIGURATIONS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

# Change DATASET_PARENT if the generated datasets are elsewhere.
DATASET_PARENT = Path(
    r"C:\Users\ASUS\Downloads"
)

# These folder names correspond to CONFIGURATION_NAME values in
# generate_gaussian_lbp_parameter_dataset.py.
STAGE1_CONFIGURATIONS = {
    "G1": {
        "dataset": (
            DATASET_PARENT /
            "pcb_dataset_gaussian_lbp_g1"
        ),
        "gaussian_kernel": 3,
        "gaussian_sigma": 0.5,
    },
    "G2": {
        "dataset": (
            DATASET_PARENT /
            "pcb_dataset_gaussian_lbp_g2"
        ),
        "gaussian_kernel": 3,
        "gaussian_sigma": 0.8,
    },
    "G3": {
        "dataset": (
            DATASET_PARENT /
            "pcb_dataset_gaussian_lbp_g3"
        ),
        "gaussian_kernel": 5,
        "gaussian_sigma": 1.0,
    },
}

# LBP settings must remain fixed during Stage 1.
FIXED_LBP_POINTS = 8
FIXED_LBP_RADIUS = 1.0
FIXED_LBP_WEIGHT = 0.25


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
# Same settings as train_student1_blocksize.py
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
    "stage1" /
    f"seed_{SEED}"
)


# =========================================================
# DATASET AND ENVIRONMENT CHECKS
# =========================================================

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

    actual_kernel = gaussian.get("kernel_size")
    actual_sigma = gaussian.get("sigma")
    actual_points = lbp.get("points")
    actual_radius = lbp.get("radius")
    actual_weight = fusion.get("lbp_weight")
    actual_gaussian_weight = fusion.get("gaussian_weight")

    if actual_kernel != settings["gaussian_kernel"]:
        raise RuntimeError(
            f"{configuration_name} Gaussian kernel mismatch: "
            f"expected {settings['gaussian_kernel']}, "
            f"found {actual_kernel}."
        )

    numeric_checks = (
        (
            "Gaussian sigma",
            actual_sigma,
            settings["gaussian_sigma"],
        ),
        ("LBP radius", actual_radius, FIXED_LBP_RADIUS),
        ("LBP weight", actual_weight, FIXED_LBP_WEIGHT),
        (
            "Gaussian weight",
            actual_gaussian_weight,
            1.0 - FIXED_LBP_WEIGHT,
        ),
    )

    for parameter, actual, expected in numeric_checks:
        if actual is None or not math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                f"{configuration_name} {parameter} mismatch: "
                f"expected {expected}, found {actual}."
            )

    if actual_points != FIXED_LBP_POINTS:
        raise RuntimeError(
            f"{configuration_name} LBP points mismatch: "
            f"expected {FIXED_LBP_POINTS}, found {actual_points}."
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
        "gaussian_kernel": actual_kernel,
        "gaussian_sigma": float(actual_sigma),
        "lbp_points": actual_points,
        "lbp_radius": float(actual_radius),
        "lbp_weight": float(actual_weight),
        "gaussian_weight": float(actual_gaussian_weight),
        "processing_seconds_per_image": seconds_per_image,
    }


def check_environment():
    print("=" * 80)
    print("STAGE 1 GAUSSIAN PARAMETER TUNING")
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
    print(f"Fixed LBP P     : {FIXED_LBP_POINTS}")
    print(f"Fixed LBP R     : {FIXED_LBP_RADIUS}")
    print(f"Fixed LBP weight: {FIXED_LBP_WEIGHT}")
    print(f"Output          : {RUNS_ROOT}")

    validated_parameters = {}

    for configuration_name, settings in (
        STAGE1_CONFIGURATIONS.items()
    ):
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
            f"kernel={parameters['gaussian_kernel']}, "
            f"sigma={parameters['gaussian_sigma']}, "
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

    # Load the same fresh pretrained model for every experiment.
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
    output_path = RUNS_ROOT / "stage1_results.csv"

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
    print("STAGE 1 GAUSSIAN PARAMETER COMPARISON")
    print("=" * 100)
    print(
        f"{'Config':<10}"
        f"{'Kernel':<10}"
        f"{'Sigma':<10}"
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
            f"{result['gaussian_kernel']:<10}"
            f"{result['gaussian_sigma']:<10.2f}"
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
    print("BEST STAGE 1 CONFIGURATION")
    print("=" * 100)
    print(f"Configuration  : {best['configuration']}")
    print(f"Gaussian kernel: {best['gaussian_kernel']}")
    print(f"Gaussian sigma : {best['gaussian_sigma']}")
    print(f"mAP50-95       : {best['map50_95']:.6f}")
    print(f"mAP50          : {best['map50']:.6f}")
    print(f"F1-score       : {best['f1_score']:.6f}")
    print(
        "Lowest class AP: "
        f"{best['lowest_class_map50_95']:.6f}"
    )

    winner_path = RUNS_ROOT / "stage1_winner.json"
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
    validated_parameters = check_environment()
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = []

    for configuration_name, settings in (
        STAGE1_CONFIGURATIONS.items()
    ):
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

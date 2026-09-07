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

MODEL_PATH = (
    PROJECT_ROOT /
    "yolov8n.pt"
)

# Change these paths if your generated folders have
# different names.
STAGE1_DATASETS = {
    "B1": Path(
        r"C:\Users\hangg\Downloads"
        r"\pcb_dataset_stage1_b1"
    ),
    "B2": Path(
        r"C:\Users\hangg\Downloads"
        r"\pcb_dataset_stage1_b2"
    ),
    "B3": Path(
        r"C:\Users\hangg\Downloads"
        r"\pcb_dataset_stage1_b3"
    ),
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
    / "stage1"
    / f"seed_{SEED}"
)


# =========================================================
# ENVIRONMENT AND DATASET CHECKS
# =========================================================

def check_environment():
    print("=" * 70)
    print("STAGE 1 BILATERAL PARAMETER TUNING")
    print("=" * 70)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the PyTorch "
            "and NVIDIA installation."
        )

    print(
        f"GPU            : "
        f"{torch.cuda.get_device_name(0)}"
    )
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

    for configuration, dataset_root in (
        STAGE1_DATASETS.items()
    ):
        yaml_path = dataset_root / "data.yaml"
        config_path = (
            dataset_root /
            "processing_config.json"
        )

        if not yaml_path.exists():
            raise FileNotFoundError(
                f"{configuration} data.yaml not found:\n"
                f"{yaml_path}"
            )

        if not config_path.exists():
            raise FileNotFoundError(
                f"{configuration} configuration not found:\n"
                f"{config_path}"
            )

        print(
            f"{configuration}: dataset found at "
            f"{dataset_root}"
        )


# =========================================================
# READ PREPROCESSING PARAMETERS
# =========================================================

def read_processing_configuration(dataset_root):
    config_path = (
        dataset_root /
        "processing_config.json"
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        configuration = json.load(file)

    bilateral = configuration["bilateral_filter"]
    morphology = configuration["morphology"]

    processing_statistics = configuration.get(
        "processing_statistics",
        [],
    )

    total_images = sum(
        item.get("images", 0)
        for item in processing_statistics
    )

    total_time = sum(
        item.get("elapsed_seconds", 0)
        for item in processing_statistics
    )

    if total_images > 0:
        seconds_per_image = total_time / total_images
    else:
        seconds_per_image = 0.0

    return {
        "bilateral_d": bilateral["diameter"],
        "sigma_color": bilateral["sigma_color"],
        "sigma_space": bilateral["sigma_space"],
        "kernel_sizes": str(
            morphology["kernel_sizes"]
        ),
        "alpha": morphology["alpha"],
        "beta": morphology["beta"],
        "processing_seconds_per_image":
            seconds_per_image,
    }


# =========================================================
# TRAIN ONE CONFIGURATION
# =========================================================

def train_configuration(
    configuration_name,
    dataset_root,
):
    data_yaml = dataset_root / "data.yaml"

    print()
    print("=" * 70)
    print(f"TRAINING CONFIGURATION: {configuration_name}")
    print(f"Dataset: {data_yaml}")
    print("=" * 70)

    # A fresh pretrained model is loaded for every
    # configuration.
    model = YOLO(
        str(MODEL_PATH)
    )

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

        # Prevent early stopping during the 50-epoch
        # comparison.
        patience=PATIENCE,

        project=str(RUNS_ROOT),
        name=f"{configuration_name}_train",
        exist_ok=False,

        save=True,
        plots=True,
        val=True,
        cache=False,
    )

    training_time = (
        time.perf_counter() - start_time
    )

    best_model_path = Path(
        model.trainer.best
    )

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"best.pt was not created:\n"
            f"{best_model_path}"
        )

    print(
        f"Best model: {best_model_path}"
    )

    return best_model_path, training_time


# =========================================================
# VALIDATE BEST MODEL
# =========================================================

def validate_configuration(
    configuration_name,
    dataset_root,
    best_model_path,
):
    data_yaml = dataset_root / "data.yaml"

    print()
    print("-" * 70)
    print(f"VALIDATING: {configuration_name}")
    print("-" * 70)

    model = YOLO(
        str(best_model_path)
    )

    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=VAL_BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,

        project=str(RUNS_ROOT),
        name=f"{configuration_name}_validation",
        exist_ok=False,

        plots=True,
    )

    precision = float(
        metrics.box.mp
    )

    recall = float(
        metrics.box.mr
    )

    map50 = float(
        metrics.box.map50
    )

    map50_95 = float(
        metrics.box.map
    )

    if precision + recall > 0:
        f1_score = (
            2 * precision * recall /
            (precision + recall)
        )
    else:
        f1_score = 0.0

    # One mAP50-95 value for each class.
    per_class_maps = list(
        metrics.box.maps
    )

    results = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "map50": map50,
        "map50_95": map50_95,
    }

    class_values = []

    for class_id, class_name in CLASS_NAMES.items():
        if class_id < len(per_class_maps):
            class_map = float(
                per_class_maps[class_id]
            )
        else:
            class_map = 0.0

        results[
            f"{class_name}_map50_95"
        ] = class_map

        class_values.append(class_map)

    results["lowest_class_map50_95"] = min(class_values)

    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1-score  : {f1_score:.6f}")
    print(f"mAP50     : {map50:.6f}")
    print(f"mAP50-95  : {map50_95:.6f}")

    return results


# =========================================================
# SAVE CSV AFTER EVERY EXPERIMENT
# =========================================================

def save_results(results):
    RUNS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RUNS_ROOT /
        "stage1_results.csv"
    )

    if not results:
        return

    fieldnames = list(
        results[0].keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nCurrent results saved to:\n"
        f"{output_path}"
    )


# =========================================================
# DISPLAY FINAL COMPARISON
# =========================================================

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
    print("=" * 90)
    print("STAGE 1 BILATERAL COMPARISON")
    print("=" * 90)

    print(
        f"{'Config':<10}"
        f"{'d':<8}"
        f"{'Sigma C':<12}"
        f"{'Sigma S':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'mAP50-95':<12}"
        f"{'Lowest AP':<12}"
    )

    print("-" * 90)

    for result in ranked:
        print(
            f"{result['configuration']:<10}"
            f"{result['bilateral_d']:<8}"
            f"{result['sigma_color']:<12}"
            f"{result['sigma_space']:<12}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1_score']:<12.4f}"
            f"{result['map50_95']:<12.4f}"
            f"{result['lowest_class_map50_95']:<12.4f}"
        )

    best = ranked[0]

    print()
    print("=" * 90)
    print("BEST STAGE 1 CONFIGURATION")
    print("=" * 90)

    print(
        f"Configuration : "
        f"{best['configuration']}"
    )

    print(
        f"Bilateral d   : "
        f"{best['bilateral_d']}"
    )

    print(
        f"Sigma colour  : "
        f"{best['sigma_color']}"
    )

    print(
        f"Sigma space   : "
        f"{best['sigma_space']}"
    )

    print(
        f"mAP50-95      : "
        f"{best['map50_95']:.6f}"
    )

    print(
        f"F1-score      : "
        f"{best['f1_score']:.6f}"
    )

    print(
        f"Lowest class AP: "
        f"{best['lowest_class_map50_95']:.6f}"
    )

    winner_path = RUNS_ROOT / "stage1_winner.json"
    winner_path.write_text(
        json.dumps(best, indent=4),
        encoding="utf-8",
    )
    print(f"Winner saved to:\n{winner_path}")

    if len(ranked) >= 2:
        difference = (
            ranked[0]["map50_95"]
            - ranked[1]["map50_95"]
        )

        print(
            f"Difference from second place: "
            f"{difference:.6f}"
        )

        if difference < 0.005:
            print(
                "The top two results are very close. "
                "Repeat both configurations using "
                "additional random seeds."
            )

    return best


# =========================================================
# MAIN
# =========================================================

def main():
    check_environment()

    RUNS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for configuration_name, dataset_root in (
        STAGE1_DATASETS.items()
    ):
        preprocessing_parameters = (
            read_processing_configuration(
                dataset_root
            )
        )

        (
            best_model_path,
            training_time,
        ) = train_configuration(
            configuration_name,
            dataset_root,
        )

        validation_results = (
            validate_configuration(
                configuration_name,
                dataset_root,
                best_model_path,
            )
        )

        experiment_result = {
            "configuration": configuration_name,
            "dataset": str(dataset_root),
            "seed": SEED,
            **preprocessing_parameters,
            **validation_results,
            "training_seconds": training_time,
            "best_model": str(best_model_path),
        }

        all_results.append(
            experiment_result
        )

        # Save after every completed configuration.
        save_results(
            all_results
        )

    print_comparison(
        all_results
    )


if __name__ == "__main__":
    main()

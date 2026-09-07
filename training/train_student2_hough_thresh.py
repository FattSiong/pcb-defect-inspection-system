from ultralytics import YOLO
from pathlib import Path
import torch
import csv
import time


# =========================================================
# GENERAL SETTINGS
# =========================================================

MODEL_NAME = "yolov8n.pt"

EPOCHS = 100
IMAGE_SIZE = 640

TRAIN_BATCH_SIZE = -1
VAL_BATCH_SIZE = 4

DEVICE = 0
SEED = 42


# =========================================================
# FIXED STUDENT 2 PARAMETERS
# =========================================================

CANNY_THRESH = "50-150"
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 10

# Hough = 50 already exists from Phase 1.
# Only create NEW training runs for 30 and 80.
HOUGH_VALUES = [
    30,
    80
]


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(
    r"D:\pcb-defect-inspection-system"
)


HOUGH50_DATASET_YAML = Path(
    r"C:\Users\user\Downloads"
    r"\pcb_dataset_student2_hough_thresh"
    r"\hough_50"
    r"\data.yaml"
)


# Existing trained Hough=50 / Canny=50-150 model
HOUGH50_MODEL_PATH = (
    PROJECT_ROOT /
    "runs" /
    "student2_canny_thresh" /
    "canny_50_150" /
    "weights" /
    "best.pt"
)


# ---------------------------------------------------------
# New Hough = 30 and Hough = 80 datasets
# ---------------------------------------------------------

HOUGH_DATASET_ROOT = Path(
    r"C:\Users\user\Downloads"
    r"\pcb_dataset_student2_hough_thresh"
)


# ---------------------------------------------------------
# Experiment output
# ---------------------------------------------------------

RUNS_ROOT = (
    PROJECT_ROOT /
    "runs" /
    "student2_hough_thresh"
)

COMPARISON_CSV = (
    RUNS_ROOT /
    "hough_thresh_comparison.csv"
)


# =========================================================
# ENVIRONMENT CHECK
# =========================================================

def check_environment():

    print("=" * 90)
    print("STUDENT 2 - HOUGH THRESHOLD PARAMETER TUNING (PHASE 2)")
    print("=" * 90)


    print(
        "PyTorch version :",
        torch.__version__
    )

    print(
        "CUDA available  :",
        torch.cuda.is_available()
    )


    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )


    print(
        "GPU             :",
        torch.cuda.get_device_name(0)
    )


    print(
        f"\nFixed Canny       : "
        f"{CANNY_THRESH}"
    )

    print(
        f"Fixed Min Length  : "
        f"{HOUGH_MIN_LINE_LENGTH}"
    )

    print(
        f"Fixed Max Gap     : "
        f"{HOUGH_MAX_LINE_GAP}"
    )

    print(
        f"Epochs            : "
        f"{EPOCHS}"
    )

    print(
        f"Hough values      : "
        f"{HOUGH_VALUES}"
    )


# =========================================================
# GET DATASET YAML
# =========================================================

def get_dataset_yaml(hough_thresh):

    # -----------------------------------------------------
    # Hough = 50 uses existing Phase 1 dataset
    # -----------------------------------------------------

    if hough_thresh == 50:

        return HOUGH50_DATASET_YAML


    # -----------------------------------------------------
    # Hough = 30 / 80 use newly processed datasets
    # -----------------------------------------------------

    return (
        HOUGH_DATASET_ROOT /
        f"hough_{hough_thresh}" /
        "data.yaml"
    )


# =========================================================
# VERIFY FILES
# =========================================================

def check_required_files():

    print("\n" + "=" * 90)
    print("FILE CHECK")
    print("=" * 90)


    if not HOUGH50_DATASET_YAML.exists():

        raise FileNotFoundError(
            f"Hough=50 dataset not found:\n"
            f"{HOUGH50_DATASET_YAML}"
        )


    if not HOUGH50_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Hough=50 existing best.pt not found:\n"
            f"{HOUGH50_MODEL_PATH}"
        )


    print(
        "\nHough=50 existing model found:"
    )

    print(
        HOUGH50_MODEL_PATH
    )


    for hough_thresh in HOUGH_VALUES:

        yaml_path = get_dataset_yaml(
            hough_thresh
        )

        if not yaml_path.exists():

            raise FileNotFoundError(
                f"\nHough={hough_thresh} dataset not found:\n"
                f"{yaml_path}\n\n"
                f"Run preprocess_student2_hough_thresh.py first."
            )


        print(
            f"\nHough={hough_thresh} dataset found:"
        )

        print(
            yaml_path
        )


# =========================================================
# TRAIN NEW CONFIGURATION
# =========================================================

def train_new_configuration(hough_thresh):

    yaml_path = get_dataset_yaml(
        hough_thresh
    )


    experiment_name = (
        f"hough_{hough_thresh}"
    )


    print("\n" + "=" * 90)

    print(
        f"TRAINING HOUGH THRESHOLD = {hough_thresh}"
    )

    print("=" * 90)


    print(
        f"Canny           : {CANNY_THRESH}"
    )

    print(
        f"Hough Threshold : {hough_thresh}"
    )

    print(
        f"Epochs          : {EPOCHS}"
    )

    model = YOLO(
        MODEL_NAME
    )


    start_time = time.time()


    model.train(

        data=str(
            yaml_path
        ),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=TRAIN_BATCH_SIZE,

        device=DEVICE,

        seed=SEED,

        patience=150,

        project=str(
            RUNS_ROOT
        ),

        name=experiment_name,

        exist_ok=True,

        plots=True
    )


    training_time = (
        time.time()
        - start_time
    )


    best_model = Path(
        model.trainer.best
    )


    print(
        "\nTraining completed."
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"\nBest model:\n"
        f"{best_model}"
    )


    return (
        yaml_path,
        best_model,
        training_time
    )


# =========================================================
# VALIDATE ONE MODEL
# =========================================================

def validate_model(
    hough_thresh,
    yaml_path,
    model_path,
    training_time=None
):

    print("\n" + "-" * 90)

    print(
        f"VALIDATING HOUGH THRESHOLD = {hough_thresh}"
    )

    print("-" * 90)


    print(
        f"Model:\n{model_path}"
    )


    model = YOLO(
        str(model_path)
    )


    metrics = model.val(

        data=str(
            yaml_path
        ),

        split="val",

        imgsz=IMAGE_SIZE,

        batch=VAL_BATCH_SIZE,

        device=DEVICE,

        project=str(
            RUNS_ROOT
        ),

        name=(
            f"hough_{hough_thresh}_validation"
        ),

        exist_ok=True,

        plots=True
    )


    # =====================================================
    # EXTRACT METRICS
    # =====================================================

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


    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0


    print(
        f"\nPrecision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    print(
        f"mAP50     : {map50:.4f}"
    )

    print(
        f"mAP50-95  : {map50_95:.4f}"
    )


    return {
        "Hough Threshold":
            hough_thresh,

        "Canny Threshold":
            CANNY_THRESH,

        "Min Line Length":
            HOUGH_MIN_LINE_LENGTH,

        "Max Line Gap":
            HOUGH_MAX_LINE_GAP,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1-score":
            f1,

        "mAP50":
            map50,

        "mAP50-95":
            map50_95,

        "Training Time (seconds)":
            (
                training_time
                if training_time is not None
                else 0.0
            ),

        "Best Model":
            str(model_path)
    }


# =========================================================
# SAVE COMPARISON CSV
# =========================================================

def save_results(results):

    RUNS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    fieldnames = [
        "Hough Threshold",
        "Canny Threshold",
        "Min Line Length",
        "Max Line Gap",
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95",
        "Training Time (seconds)",
        "Best Model"
    ]


    with open(
        COMPARISON_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    print(
        f"\nComparison saved to:\n"
        f"{COMPARISON_CSV}"
    )


# =========================================================
# PRINT FINAL COMPARISON
# =========================================================

def print_comparison(results):

    results = sorted(
        results,
        key=lambda item:
            item["Hough Threshold"]
    )


    print("\n" + "=" * 110)
    print("HOUGH THRESHOLD VALIDATION COMPARISON")
    print("=" * 110)


    print(
        f"{'Hough':<8}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
        f"{'F1':<14}"
        f"{'mAP50':<14}"
        f"{'mAP50-95':<14}"
    )


    print(
        "-" * 110
    )


    for result in results:

        print(
            f"{result['Hough Threshold']:<8}"
            f"{result['Precision']:<14.4f}"
            f"{result['Recall']:<14.4f}"
            f"{result['F1-score']:<14.4f}"
            f"{result['mAP50']:<14.4f}"
            f"{result['mAP50-95']:<14.4f}"
        )


    # =====================================================
    # FIND BEST HOUGH USING mAP50-95
    # =====================================================

    best = max(
        results,
        key=lambda result:
            result["mAP50-95"]
    )


    print("\n" + "=" * 110)
    print("BEST HOUGH THRESHOLD VALUE")
    print("=" * 110)


    print(
        f"Hough Threshold : "
        f"{best['Hough Threshold']}"
    )

    print(
        f"Canny Threshold : "
        f"{best['Canny Threshold']}"
    )

    print(
        f"Precision       : "
        f"{best['Precision']:.4f}"
    )

    print(
        f"Recall          : "
        f"{best['Recall']:.4f}"
    )

    print(
        f"F1-score        : "
        f"{best['F1-score']:.4f}"
    )

    print(
        f"mAP50           : "
        f"{best['mAP50']:.4f}"
    )

    print(
        f"mAP50-95        : "
        f"{best['mAP50-95']:.4f}"
    )


    print(
        "\nBest model:"
    )

    print(
        best["Best Model"]
    )


# =========================================================
# MAIN
# =========================================================

def main():

    check_environment()

    check_required_files()


    results = []


    print("\n" + "=" * 90)
    print("REUSING EXISTING HOUGH = 50 MODEL")
    print("=" * 90)


    hough50_result = validate_model(
        hough_thresh=50,
        yaml_path=HOUGH50_DATASET_YAML,
        model_path=HOUGH50_MODEL_PATH,
        training_time=None
    )


    results.append(
        hough50_result
    )



    save_results(
        results
    )


    # =====================================================
    # HOUGH = 30 AND HOUGH = 80
    # =====================================================

    for hough_thresh in HOUGH_VALUES:

        (
            yaml_path,
            best_model,
            training_time
        ) = train_new_configuration(
            hough_thresh
        )


        result = validate_model(
            hough_thresh,
            yaml_path,
            best_model,
            training_time
        )


        results.append(
            result
        )



        save_results(
            results
        )


    # =====================================================
    # FINAL COMPARISON
    # =====================================================

    print_comparison(
        results
    )


    print("\n" + "=" * 110)
    print("HOUGH PARAMETER TUNING COMPLETED")
    print("=" * 110)


if __name__ == "__main__":
    main()
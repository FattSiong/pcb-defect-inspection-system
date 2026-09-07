from ultralytics import YOLO
from pathlib import Path
import torch
import csv
import time


# =========================================================
# YOLO SETTINGS
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
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 60

# maxLineGap = 10 already exists from Phase 3
# Only train NEW datasets
MAX_LINE_GAPS = [
    2,
    5
]


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(
    r"D:\pcb-defect-inspection-system"
)

NEW_DATASET_ROOT = Path(
    r"C:\Users\user\Downloads"
    r"\pcb_dataset_student2_max_gap"
)

RUNS_ROOT = (
    PROJECT_ROOT /
    "runs" /
    "student2_max_gap"
)

COMPARISON_CSV = (
    RUNS_ROOT /
    "max_gap_comparison.csv"
)


# =========================================================
# EXISTING maxLineGap = 10
# =========================================================

# Exact configuration from Phase 3:
# Canny = 50-150
# Hough = 80
# minLine = 60
# maxGap = 10

EXISTING_GAP10_DATASET = Path(
    r"C:\Users\user\Downloads"
    r"\pcb_dataset_student2_min_line"
    r"\min_len_60"
    r"\data.yaml"
)

EXISTING_GAP10_MODEL = (
    PROJECT_ROOT /
    "runs" /
    "student2_min_line" /
    "min_len_60" /
    "weights" /
    "best.pt"
)


# =========================================================
# CHECK ENVIRONMENT
# =========================================================

def check_environment():

    print("=" * 90)
    print("STUDENT 2 - MAX LINE GAP TUNING (PHASE 4)")
    print("=" * 90)

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA:",
        torch.cuda.is_available()
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        f"\nCanny Threshold  : "
        f"{CANNY_THRESH}"
    )

    print(
        f"Hough Threshold  : "
        f"{HOUGH_THRESHOLD}"
    )

    print(
        f"Min Line Length  : "
        f"{HOUGH_MIN_LINE_LENGTH}"
    )

    print(
        f"maxLineGaps      : "
        f"{MAX_LINE_GAPS}"
    )

    print(
        f"Epochs           : "
        f"{EPOCHS}"
    )


# =========================================================
# GET DATASET
# =========================================================

def get_dataset_yaml(
    max_gap
):

    # Existing 10 configuration
    if max_gap == 10:

        return (
            EXISTING_GAP10_DATASET
        )


    return (
        NEW_DATASET_ROOT /
        f"max_gap_{max_gap}" /
        "data.yaml"
    )


# =========================================================
# CHECK FILES
# =========================================================

def check_required_files():

    print("\n" + "=" * 90)
    print("CHECKING FILES")
    print("=" * 90)


    if not EXISTING_GAP10_DATASET.exists():

        raise FileNotFoundError(
            f"Existing maxLineGap=10 "
            f"dataset not found:\n"
            f"{EXISTING_GAP10_DATASET}"
        )


    if not EXISTING_GAP10_MODEL.exists():

        raise FileNotFoundError(
            f"Existing maxLineGap=10 "
            f"best.pt not found:\n"
            f"{EXISTING_GAP10_MODEL}"
        )


    print(
        "\nmaxLineGap=10 model:"
    )

    print(
        EXISTING_GAP10_MODEL
    )


    for max_gap in [
        2,
        5
    ]:

        yaml_path = get_dataset_yaml(
            max_gap
        )

        if not yaml_path.exists():

            raise FileNotFoundError(
                f"\nmaxLineGap="
                f"{max_gap} "
                f"dataset missing:\n"
                f"{yaml_path}\n\n"
                f"Run preprocessing first."
            )

        print(
            f"\nmaxLineGap="
            f"{max_gap}:"
        )

        print(
            yaml_path
        )


# =========================================================
# TRAIN NEW MODEL
# =========================================================

def train_configuration(
    max_gap
):

    yaml_path = get_dataset_yaml(
        max_gap
    )

    experiment_name = (
        f"max_gap_{max_gap}"
    )


    print("\n" + "=" * 90)

    print(
        f"TRAINING maxLineGap = "
        f"{max_gap}"
    )

    print("=" * 90)


    # Fresh YOLOv8n for fair comparison
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


    if not best_model.exists():

        raise FileNotFoundError(
            f"Training completed but best.pt "
            f"was not found:\n"
            f"{best_model}"
        )


    print(
        f"\nTraining time: "
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
# VALIDATE
# =========================================================

def validate_model(
    max_gap,
    yaml_path,
    model_path,
    training_time=0.0
):

    print("\n" + "-" * 90)

    print(
        f"VALIDATING maxLineGap = "
        f"{max_gap}"
    )

    print("-" * 90)


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
            f"max_gap_{max_gap}"
            f"_validation"
        ),

        exist_ok=True,

        plots=True
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


    if (
        precision
        + recall
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
        f"\nPrecision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1-score  : "
        f"{f1:.4f}"
    )

    print(
        f"mAP50     : "
        f"{map50:.4f}"
    )

    print(
        f"mAP50-95  : "
        f"{map50_95:.4f}"
    )


    return {
        "Max Line Gap":
            max_gap,

        "Min Line Length":
            HOUGH_MIN_LINE_LENGTH,

        "Canny Threshold":
            CANNY_THRESH,

        "Hough Threshold":
            HOUGH_THRESHOLD,

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
            training_time,

        "Best Model":
            str(model_path)
    }


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(
    results
):

    RUNS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    results = sorted(
        results,
        key=lambda x:
            x["Max Line Gap"]
    )


    fieldnames = [
        "Max Line Gap",
        "Min Line Length",
        "Canny Threshold",
        "Hough Threshold",
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
        f"\nResults saved:\n"
        f"{COMPARISON_CSV}"
    )


# =========================================================
# PRINT COMPARISON
# =========================================================

def print_comparison(
    results
):

    results = sorted(
        results,
        key=lambda x:
            x["Max Line Gap"]
    )


    print("\n" + "=" * 105)
    print("MAX LINE GAP VALIDATION COMPARISON")
    print("=" * 105)


    print(
        f"{'Max Gap':<14}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
        f"{'F1':<14}"
        f"{'mAP50':<14}"
        f"{'mAP50-95':<14}"
    )

    print(
        "-" * 105
    )


    for result in results:

        print(
            f"{result['Max Line Gap']:<14}"
            f"{result['Precision']:<14.4f}"
            f"{result['Recall']:<14.4f}"
            f"{result['F1-score']:<14.4f}"
            f"{result['mAP50']:<14.4f}"
            f"{result['mAP50-95']:<14.4f}"
        )


    best = max(
        results,
        key=lambda x:
            x["mAP50-95"]
    )


    print("\n" + "=" * 105)
    print("BEST MAX LINE GAP")
    print("=" * 105)

    print(
        f"Max Gap    : "
        f"{best['Max Line Gap']}"
    )

    print(
        f"Min Length : "
        f"{best['Min Line Length']}"
    )

    print(
        f"Canny      : "
        f"{best['Canny Threshold']}"
    )

    print(
        f"Hough Thr  : "
        f"{best['Hough Threshold']}"
    )

    print(
        f"Precision  : "
        f"{best['Precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{best['Recall']:.4f}"
    )

    print(
        f"F1-score   : "
        f"{best['F1-score']:.4f}"
    )

    print(
        f"mAP50      : "
        f"{best['mAP50']:.4f}"
    )

    print(
        f"mAP50-95   : "
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


    # =====================================================
    # maxLineGap = 10
    # REUSE EXISTING MODEL
    # =====================================================

    print("\n" + "=" * 90)
    print("REUSING EXISTING maxLineGap = 10 MODEL")
    print("=" * 90)

    existing_result = validate_model(

        max_gap=10,

        yaml_path=EXISTING_GAP10_DATASET,

        model_path=EXISTING_GAP10_MODEL,

        training_time=0.0
    )

    results.append(
        existing_result
    )

    save_results(
        results
    )


    # =====================================================
    # NEW maxLineGap 2 AND 5
    # =====================================================

    for max_gap in [
        2,
        5
    ]:

        (
            yaml_path,
            best_model,
            training_time
        ) = train_configuration(
            max_gap
        )


        result = validate_model(
            max_gap,
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


    print("\n" + "=" * 105)
    print("MAX LINE GAP PARAMETER TUNING COMPLETED")
    print("=" * 105)


if __name__ == "__main__":
    main()
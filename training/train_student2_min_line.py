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
HOUGH_MAX_LINE_GAP = 10

MIN_LINE_LENGTHS = [
    20,
    40,
    60
]


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(
    r"D:\pcb-defect-inspection-system"
)

NEW_DATASET_ROOT = Path(
    r"C:\Users\user\Downloads"
    r"\pcb_dataset_student2_min_line"
)

RUNS_ROOT = (
    PROJECT_ROOT /
    "runs" /
    "student2_min_line"
)

COMPARISON_CSV = (
    RUNS_ROOT /
    "min_line_comparison.csv"
)


# =========================================================
# EXISTING minLineLength = 20
# =========================================================


EXISTING_MIN20_DATASET = Path(
    r"C:\Users\user\Downloads"
    r"\pcb_dataset_student2_hough_thresh"
    r"\hough_80"
    r"\data.yaml"
)

EXISTING_MIN20_MODEL = (
    PROJECT_ROOT /
    "runs" /
    "student2_hough_thresh" /
    "hough_80" /
    "weights" /
    "best.pt"
)


# =========================================================
# CHECK ENVIRONMENT
# =========================================================

def check_environment():

    print("=" * 90)
    print("STUDENT 2 - MINIMUM LINE LENGTH TUNING (PHASE 3)")
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
        f"Max Line Gap     : "
        f"{HOUGH_MAX_LINE_GAP}"
    )

    print(
        f"minLineLengths   : "
        f"{MIN_LINE_LENGTHS}"
    )

    print(
        f"Epochs           : "
        f"{EPOCHS}"
    )


# =========================================================
# GET DATASET
# =========================================================

def get_dataset_yaml(
    min_line
):

    if min_line == 20:

        return (
            EXISTING_MIN20_DATASET
        )


    return (
        NEW_DATASET_ROOT /
        f"min_len_{min_line}" /
        "data.yaml"
    )


# =========================================================
# CHECK FILES
# =========================================================

def check_required_files():

    print("\n" + "=" * 90)
    print("CHECKING FILES")
    print("=" * 90)


    if not EXISTING_MIN20_DATASET.exists():

        raise FileNotFoundError(
            f"Existing minLineLength=20 "
            f"dataset not found:\n"
            f"{EXISTING_MIN20_DATASET}"
        )


    if not EXISTING_MIN20_MODEL.exists():

        raise FileNotFoundError(
            f"Existing minLineLength=20 "
            f"best.pt not found:\n"
            f"{EXISTING_MIN20_MODEL}"
        )


    print(
        "\nminLineLength=20 model:"
    )

    print(
        EXISTING_MIN20_MODEL
    )


    for min_line in [
        40,
        60
    ]:

        yaml_path = get_dataset_yaml(
            min_line
        )

        if not yaml_path.exists():

            raise FileNotFoundError(
                f"\nminLineLength="
                f"{min_line} "
                f"dataset missing:\n"
                f"{yaml_path}\n\n"
                f"Run preprocessing first."
            )

        print(
            f"\nminLineLength="
            f"{min_line}:"
        )

        print(
            yaml_path
        )


# =========================================================
# TRAIN NEW MODEL
# =========================================================

def train_configuration(
    min_line
):

    yaml_path = get_dataset_yaml(
        min_line
    )

    experiment_name = (
        f"min_len_{min_line}"
    )


    print("\n" + "=" * 90)

    print(
        f"TRAINING minLineLength = "
        f"{min_line}"
    )

    print("=" * 90)


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
    min_line,
    yaml_path,
    model_path,
    training_time=0.0
):

    print("\n" + "-" * 90)

    print(
        f"VALIDATING minLineLength = "
        f"{min_line}"
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
            f"min_len_{min_line}"
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
        "Min Line Length":
            min_line,

        "Canny Threshold":
            CANNY_THRESH,

        "Hough Threshold":
            HOUGH_THRESHOLD,

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
            x["Min Line Length"]
    )


    fieldnames = [
        "Min Line Length",
        "Canny Threshold",
        "Hough Threshold",
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
            x["Min Line Length"]
    )


    print("\n" + "=" * 105)
    print("MIN LINE LENGTH VALIDATION COMPARISON")
    print("=" * 105)


    print(
        f"{'Min Length':<14}"
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
            f"{result['Min Line Length']:<14}"
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
    print("BEST MIN LINE LENGTH")
    print("=" * 105)

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


    print("\n" + "=" * 90)
    print("REUSING EXISTING minLineLength = 20 MODEL")
    print("=" * 90)

    existing_result = validate_model(

        min_line=20,

        yaml_path=EXISTING_MIN20_DATASET,

        model_path=EXISTING_MIN20_MODEL,

        training_time=0.0
    )

    results.append(
        existing_result
    )

    save_results(
        results
    )


    # =====================================================
    # NEW minLineLength 40 AND 60
    # =====================================================

    for min_line in [
        40,
        60
    ]:

        (
            yaml_path,
            best_model,
            training_time
        ) = train_configuration(
            min_line
        )


        result = validate_model(
            min_line,
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
    print("MIN LINE LENGTH PARAMETER TUNING COMPLETED")
    print("=" * 105)


if __name__ == "__main__":
    main()
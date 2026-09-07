from ultralytics import YOLO
from pathlib import Path
import torch
import csv
import time


# =========================================================
# SETTINGS
# =========================================================

MODEL_NAME = "yolov8n.pt"

PROCESSED_ROOT = Path(
    r"C:\Users\user\Downloads\pcb_dataset_student2_canny_thresh"
)

RUNS_ROOT = Path(
    r"D:\pcb-defect-inspection-system"
    r"\runs\student2_canny_thresh"
)


# =========================================================
# PARAMETERS BEING COMPARED (PHASE 1)
# =========================================================

CANNY_THRESHOLDS = [
    (50, 150),
    (100, 200),
    (150, 250)
]


# =========================================================
# FIXED YOLO SETTINGS
# =========================================================

EPOCHS = 100
IMAGE_SIZE = 640

TRAIN_BATCH_SIZE = -1

# model.val() needs positive integer
VAL_BATCH_SIZE = 4

DEVICE = 0
SEED = 42


# =========================================================
# CHECK ENVIRONMENT
# =========================================================

def check_environment():

    print("=" * 80)
    print("STUDENT 2 CANNY THRESHOLD TRAINING (PHASE 1)")
    print("=" * 80)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Canny Thresholds:",
        CANNY_THRESHOLDS
    )


# =========================================================
# TRAIN ONE CONFIGURATION
# =========================================================

def train_canny_thresh(
    canny_thresh
):

    folder_name = (
        f"canny_{canny_thresh[0]}_{canny_thresh[1]}"
    )

    dataset = (
        PROCESSED_ROOT /
        folder_name
    )

    yaml_path = (
        dataset /
        "data.yaml"
    )

    if not yaml_path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{yaml_path}\n\n"
            f"Run preprocessing first."
        )

    experiment_name = folder_name

    print("\n" + "=" * 80)
    print(
        f"TRAINING CANNY THRESHOLD {canny_thresh}"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # Fresh pretrained model for every threshold configuration
    # -----------------------------------------------------

    model = YOLO(
        MODEL_NAME
    )

    start_time = time.time()

    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=TRAIN_BATCH_SIZE,
        device=DEVICE,
        seed=SEED,
        patience=150,
        project=str(RUNS_ROOT),
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
        "\nBest model:"
    )

    print(
        best_model
    )

    return (
        yaml_path,
        best_model,
        training_time
    )


# =========================================================
# VALIDATE BEST MODEL
# =========================================================

def validate_model(
    canny_thresh,
    yaml_path,
    best_model,
    training_time
):

    folder_name = (
        f"canny_{canny_thresh[0]}_{canny_thresh[1]}"
    )

    print("\n" + "-" * 80)
    print(
        f"VALIDATING CANNY THRESHOLD {canny_thresh}"
    )
    print("-" * 80)

    model = YOLO(
        str(best_model)
    )

    metrics = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=VAL_BATCH_SIZE,
        device=DEVICE,
        project=str(RUNS_ROOT),
        name=(
            f"{folder_name}_validation"
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
        f"Precision : {precision:.4f}"
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
        "Canny Threshold":
            f"{canny_thresh[0]}-{canny_thresh[1]}",

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
            str(best_model)
    }


# =========================================================
# SAVE COMPARISON
# =========================================================

def save_results(results):

    RUNS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        RUNS_ROOT /
        "canny_thresh_comparison.csv"
    )

    fieldnames = [
        "Canny Threshold",
        "Precision",
        "Recall",
        "F1-score",
        "mAP50",
        "mAP50-95",
        "Training Time (seconds)",
        "Best Model"
    ]

    with open(
        output_file,
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
        f"{output_file}"
    )


# =========================================================
# PRINT FINAL COMPARISON
# =========================================================

def print_comparison(results):

    print("\n" + "=" * 95)
    print("CANNY THRESHOLD VALIDATION COMPARISON")
    print("=" * 95)

    print(
        f"{'Threshold':<16}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'mAP50':<12}"
        f"{'mAP50-95':<12}"
    )

    print(
        "-" * 95
    )

    for result in results:

        print(
            f"{result['Canny Threshold']:<16}"
            f"{result['Precision']:<12.4f}"
            f"{result['Recall']:<12.4f}"
            f"{result['F1-score']:<12.4f}"
            f"{result['mAP50']:<12.4f}"
            f"{result['mAP50-95']:<12.4f}"
        )


    best = max(
        results,
        key=lambda result:
            result["mAP50-95"]
    )

    print("\n" + "=" * 95)
    print("BEST CANNY THRESHOLD CONFIGURATION")
    print("=" * 95)

    print(
        f"Canny Threshold : "
        f"{best['Canny Threshold']}"
    )

    print(
        f"mAP50-95        : "
        f"{best['mAP50-95']:.4f}"
    )

    print(
        f"mAP50           : "
        f"{best['mAP50']:.4f}"
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
        "\nBest model weights:"
    )

    print(
        best["Best Model"]
    )


# =========================================================
# MAIN
# =========================================================

def main():

    check_environment()

    all_results = []

    for canny_thresh in CANNY_THRESHOLDS:

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        (
            yaml_path,
            best_model,
            training_time
        ) = train_canny_thresh(
            canny_thresh
        )

        # -------------------------------------------------
        # Validate
        # -------------------------------------------------

        result = validate_model(
            canny_thresh,
            yaml_path,
            best_model,
            training_time
        )

        all_results.append(
            result
        )

        # Save after every run
        save_results(
            all_results
        )

    # -----------------------------------------------------
    # Final comparison
    # -----------------------------------------------------

    print_comparison(
        all_results
    )


if __name__ == "__main__":
    main()
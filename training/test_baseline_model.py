from ultralytics import YOLO
from pathlib import Path
import torch
import csv
import json


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system"
    r"\runs\epoch_testing\reduced_dataset_100_epochs"
    r"\weights\best.pt"
)

DATASET = Path(
    r"C:\Users\tiong\Downloads\pcb_dataset_reduced\data.yaml"
)

IMAGE_SIZE = 640

# model.val() requires a positive batch size
BATCH_SIZE = 4

DEVICE = 0

OUTPUT_DIR = Path(
    r"C:\Users\tiong\pcb-defect-inspection-system\runs\baseline"
)

TEST_NAME = "original_images_test"

RESULT_DIR = OUTPUT_DIR / TEST_NAME


# =========================================================
# CHECK FILES AND ENVIRONMENT
# =========================================================

def check_files():

    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATASET}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    print("GPU     :", torch.cuda.get_device_name(0))
    print("Model   :", MODEL_PATH)
    print("Dataset :", DATASET)


# =========================================================
# SAVE OVERALL METRICS
# =========================================================

def save_overall_metrics(
    precision,
    recall,
    f1,
    map50,
    map50_95
):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        RESULT_DIR /
        "test_metrics_summary.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Method",
                "Precision",
                "Recall",
                "F1-score",
                "mAP50",
                "mAP50-95"
            ]
        )

        writer.writerow(
            [
                "Original Images (Baseline)",
                precision,
                recall,
                f1,
                map50,
                map50_95
            ]
        )

    print(
        f"\nOverall metrics saved:\n{csv_path}"
    )


# =========================================================
# SAVE JSON RESULT
# =========================================================

def save_json_results(
    precision,
    recall,
    f1,
    map50,
    map50_95
):

    result = {
        "method": "Original Images (Baseline)",
        "model": "YOLOv8n",
        "epochs": 100,
        "image_size": IMAGE_SIZE,
        "dataset": str(DATASET),

        "test_metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "mAP50": float(map50),
            "mAP50_95": float(map50_95)
        }
    }

    json_path = (
        RESULT_DIR /
        "test_metrics_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )

    print(
        f"\nJSON results saved:\n{json_path}"
    )


# =========================================================
# SAVE PER-CLASS RESULTS
# =========================================================

def save_class_results(
    metrics,
    model
):

    csv_path = (
        RESULT_DIR /
        "test_metrics_by_class.csv"
    )

    precision = metrics.box.p
    recall = metrics.box.r
    ap50 = metrics.box.ap50
    map50_95 = metrics.box.maps

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Class ID",
                "Defect Class",
                "Precision",
                "Recall",
                "F1-score",
                "mAP50",
                "mAP50-95"
            ]
        )

        for class_id in range(
            len(precision)
        ):

            p = float(
                precision[class_id]
            )

            r = float(
                recall[class_id]
            )

            if (p + r) > 0:

                f1 = (
                    2 * p * r /
                    (p + r)
                )

            else:
                f1 = 0.0

            writer.writerow(
                [
                    class_id,
                    model.names[class_id],
                    p,
                    r,
                    f1,
                    float(
                        ap50[class_id]
                    ),
                    float(
                        map50_95[class_id]
                    )
                ]
            )

    print(
        f"\nPer-class metrics saved:\n{csv_path}"
    )


# =========================================================
# SAVE PROCESSING SPEED
# =========================================================

def save_speed(metrics):

    csv_path = (
        RESULT_DIR /
        "test_processing_speed.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Stage",
                "Time (ms/image)"
            ]
        )

        for stage, value in (
            metrics.speed.items()
        ):

            writer.writerow(
                [
                    stage,
                    value
                ]
            )

    print(
        f"\nProcessing speed saved:\n{csv_path}"
    )


# =========================================================
# TEST MODEL
# =========================================================

def test_model():

    print("\n" + "=" * 70)
    print("BASELINE TEST EVALUATION")
    print("=" * 70)

    print(
        "Loading existing best.pt..."
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    print(
        "\nRunning model on TEST dataset..."
    )

    metrics = model.val(

        data=str(DATASET),

        # Use TEST, not validation
        split="test",

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        device=DEVICE,

        project=str(OUTPUT_DIR),

        name=TEST_NAME,

        # Keep exact directory name
        exist_ok=True,

        # Automatically generate
        # confusion matrix and curves
        plots=True
    )


    # =====================================================
    # EXTRACT OVERALL TEST METRICS
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


    if (precision + recall) > 0:

        f1 = (
            2 * precision * recall /
            (precision + recall)
        )

    else:

        f1 = 0.0


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL BASELINE TEST RESULTS")
    print("=" * 70)

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


    # =====================================================
    # PROCESSING SPEED
    # =====================================================

    print("\n" + "=" * 70)
    print("PROCESSING SPEED")
    print("=" * 70)

    for key, value in (
        metrics.speed.items()
    ):

        print(
            f"{key:<15}: "
            f"{value:.3f} ms/image"
        )


    # =====================================================
    # SAVE ALL NUMERICAL RESULTS
    # =====================================================

    save_overall_metrics(
        precision,
        recall,
        f1,
        map50,
        map50_95
    )

    save_json_results(
        precision,
        recall,
        f1,
        map50,
        map50_95
    )

    save_class_results(
        metrics,
        model
    )

    save_speed(
        metrics
    )


    # =====================================================
    # FINISHED
    # =====================================================

    print("\n" + "=" * 70)
    print("BASELINE TEST COMPLETED")
    print("=" * 70)

    print(
        f"\nAll results saved to:\n"
        f"{RESULT_DIR}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    check_files()

    test_model()


if __name__ == "__main__":
    main()